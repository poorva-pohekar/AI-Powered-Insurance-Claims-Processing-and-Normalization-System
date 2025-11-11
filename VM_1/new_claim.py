# claims_normalizer.py  (replace /mnt/data/claims_normalizer.py with this)
"""
Improved Claims Normalizer
- Rule-based extraction (keywords + regex)
- Optional spaCy NER for robust entity extraction (fallback to regex)
- Optional ML classifier (TF-IDF + LogisticRegression) to predict loss_type and severity
- Evaluation helpers
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import re
import json
import logging

# ML libs (optional)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Optional spaCy (if installed)
try:
    import spacy
    _SPACY = spacy.load("en_core_web_sm")
except Exception:
    _SPACY = None

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LossType(Enum):
    WATER_DAMAGE = "Water Damage"
    FIRE_DAMAGE = "Fire Damage"
    THEFT = "Theft"
    VANDALISM = "Vandalism"
    COLLISION = "Collision"
    STORM_DAMAGE = "Storm Damage"
    WIND_DAMAGE = "Wind Damage"
    HAIL_DAMAGE = "Hail Damage"
    LIGHTNING = "Lightning"
    EARTHQUAKE = "Earthquake"
    FLOOD = "Flood"
    SMOKE_DAMAGE = "Smoke Damage"
    ELECTRICAL_DAMAGE = "Electrical Damage"
    FROZEN_PIPE = "Frozen Pipe"
    LIABILITY = "Liability"
    MEDICAL = "Medical"
    OTHER = "Other"


class Severity(Enum):
    MINOR = "Minor"
    MODERATE = "Moderate"
    MAJOR = "Major"
    CATASTROPHIC = "Catastrophic"


@dataclass
class NormalizedClaim:
    loss_types: List[str]
    severity: str
    affected_assets: List[str]
    confidence_score: float
    extracted_entities: Dict[str, List[str]]
    raw_text: str
    predicted_loss: Optional[str] = None
    predicted_severity: Optional[str] = None
    # Add a method to convert to serializable dict
    def to_dict(self):
        return asdict(self)


class ClaimsNormalizer:
    def __init__(self, use_spacy: bool = True, use_ml: bool = False):
        self.use_spacy = use_spacy and (_SPACY is not None)
        self.use_ml = use_ml
        self._initialize_patterns()
        self._initialize_keywords()
        # ML placeholders
        self.loss_encoder = LabelEncoder()
        self.sev_encoder = LabelEncoder()
        self.loss_pipeline = None
        self.sev_pipeline = None

    def _initialize_patterns(self):
        self.patterns = {
            'monetary': r'\$[\d,]+(?:\.\d{2})?|\d+\s*(?:dollars?|USD)',
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'time': r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
            'location': r'\b(?:room|kitchen|bedroom|bathroom|garage|basement|attic|living room|dining room|hallway|porch|yard|driveway)\b',
            'vehicle': r'\b(?:\d{4}\s+)?(?:car|truck|van|suv|sedan|vehicle|auto|motorcycle|bike)\b',
        }

    def _initialize_keywords(self):
        # same keyword dictionaries as your original file but condensed for brevity
        self.loss_type_keywords = {
            LossType.WATER_DAMAGE: ['water', 'leak', 'pipe', 'plumbing', 'flood', 'moisture', 'burst pipe'],
            LossType.FIRE_DAMAGE: ['fire', 'burn', 'smoke', 'blaze', 'burnt', 'ignite'],
            LossType.THEFT: ['theft', 'stolen', 'burglary', 'robbery', 'break-in', 'stole'],
            LossType.VANDALISM: ['vandalism', 'vandalized', 'graffiti', 'defaced'],
            LossType.COLLISION: ['collision', 'crash', 'accident', 'rear-end', 'side-swipe', 'totaled'],
            LossType.STORM_DAMAGE: ['storm', 'hurricane', 'tornado', 'severe weather', 'tree fell'],
            LossType.WIND_DAMAGE: ['wind', 'windstorm', 'gust', 'blown', 'roof damage'],
            LossType.HAIL_DAMAGE: ['hail', 'hailstorm', 'hailstone'],
            LossType.LIGHTNING: ['lightning', 'lightning strike'],
            LossType.EARTHQUAKE: ['earthquake', 'seismic', 'tremor', 'quake'],
            LossType.FLOOD: ['flood', 'flooding', 'flash flood', 'submerged'],
            LossType.SMOKE_DAMAGE: ['smoke', 'soot', 'smoky'],
            LossType.ELECTRICAL_DAMAGE: ['electrical', 'power surge', 'short circuit', 'wiring'],
            LossType.FROZEN_PIPE: ['frozen', 'frozen pipe', 'freeze'],
            LossType.LIABILITY: ['liability', 'injury', 'slip', 'fall', 'personal injury'],
            LossType.MEDICAL: ['medical', 'hospital', 'ambulance', 'doctor', 'injury']
        }

        self.severity_keywords = {
            Severity.MINOR: ['minor', 'small', 'slight', 'cosmetic', 'scratch', 'dent'],
            Severity.MODERATE: ['moderate', 'considerable', 'significant', 'damaged', 'broken'],
            Severity.MAJOR: ['major', 'severe', 'extensive', 'substantial', 'large', 'destroyed'],
            Severity.CATASTROPHIC: ['catastrophic', 'total loss', 'totaled', 'complete destruction', 'uninhabitable']
        }

        self.asset_keywords = {
            'Property': ['house', 'home', 'property', 'building', 'residence'],
            'Vehicle': ['car', 'truck', 'vehicle', 'auto', 'van', 'suv', 'motorcycle'],
            'Roof': ['roof', 'shingles', 'roofing', 'ceiling'],
            'Window': ['window', 'glass', 'windshield'],
            'Door': ['door', 'entrance', 'doorway'],
            'Appliance': ['appliance', 'refrigerator', 'washer', 'dryer', 'dishwasher', 'stove', 'oven'],
            'Furniture': ['furniture', 'couch', 'sofa', 'table', 'chair', 'bed', 'desk'],
            'Electronics': ['tv', 'television', 'computer', 'laptop', 'phone', 'electronics'],
            'Flooring': ['floor', 'flooring', 'carpet', 'hardwood', 'tile'],
            'Wall': ['wall', 'drywall', 'paint', 'wallpaper'],
            'Personal Property': ['jewelry', 'clothes', 'clothing', 'belongings', 'items']
        }

    # ---------- Entity extraction ----------
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {}
        # Use spaCy NER where available
        if self.use_spacy:
            doc = _SPACY(text)
            for ent in doc.ents:
                entities.setdefault(ent.label_, []).append(ent.text)
        # Regex-based patterns
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                entities.setdefault(name, []).extend([m.strip() for m in matches if m])
        # dedupe
        for k in list(entities.keys()):
            entities[k] = sorted(list(set(entities[k])))
        return entities

    # ---------- Rule-based loss detection ----------
    def _detect_loss_types_rule(self, text: str) -> List[LossType]:
        scores = {}
        for loss_type, keywords in self.loss_type_keywords.items():
            # count occurrences (simple)
            score = sum(text.lower().count(k) for k in keywords)
            if score > 0:
                scores[loss_type] = score
        if not scores:
            return [LossType.OTHER]
        max_score = max(scores.values())
        # pick types with >= 70% of max
        picks = [lt for lt, sc in scores.items() if sc >= max_score * 0.7]
        return picks or [LossType.OTHER]

    # ---------- Severity ----------
    def _determine_severity_rule(self, text: str, entities: Dict[str, List[str]]) -> Severity:
        scores = {s: 0 for s in Severity}
        for s, keywords in self.severity_keywords.items():
            for k in keywords:
                if k in text.lower():
                    scores[s] += 2
        # monetary adjust
        if 'monetary' in entities:
            for m in entities['monetary']:
                amt = self._extract_amount(m)
                if amt is None:
                    continue
                if amt < 1000:
                    scores[Severity.MINOR] += 1
                elif amt < 5000:
                    scores[Severity.MODERATE] += 1
                elif amt < 20000:
                    scores[Severity.MAJOR] += 1
                else:
                    scores[Severity.CATASTROPHIC] += 1
        # extreme words
        if any(w in text.lower() for w in ['total', 'complete', 'entire', 'uninhabitable']):
            scores[Severity.MAJOR] += 2
            scores[Severity.CATASTROPHIC] += 1
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else Severity.MODERATE

    # ---------- Assets ----------
    def _extract_assets(self, text: str) -> List[str]:
        found = set()
        tl = text.lower()
        for asset_name, kws in self.asset_keywords.items():
            if any(k in tl for k in kws):
                found.add(asset_name)
        return sorted(list(found))

    def _extract_amount(self, value_str: str) -> Optional[float]:
        try:
            nums = re.findall(r'[\d,]+(?:\.\d{1,2})?', value_str)
            if not nums:
                return None
            return float(nums[0].replace(',', ''))
        except Exception:
            return None

    def _calculate_confidence(self,
                              loss_types: List[LossType],
                              severity: Severity,
                              assets: List[str],
                              entities: Dict[str, List[str]]) -> float:
        conf = 0.0
        if loss_types and loss_types[0] != LossType.OTHER:
            conf += 0.35
        if severity != Severity.MODERATE:
            conf += 0.2
        if assets:
            conf += min(0.3, 0.08 * len(assets))
        if entities:
            conf += min(0.15, 0.03 * len(entities))
        return round(min(conf, 1.0), 2)

    # ---------- ML training (optional) ----------
    def train_ml(self, texts: List[str], loss_labels: List[str], severity_labels: List[str]) -> None:
        """
        Train two separate text classifiers (loss type and severity).
        Labels should be strings matching LossType.value and Severity.value.
        """
        # Loss encoder
        self.loss_encoder.fit(loss_labels)
        self.sev_encoder.fit(severity_labels)

        self.loss_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        self.loss_pipeline.fit(texts, self.loss_encoder.transform(loss_labels))

        self.sev_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        self.sev_pipeline.fit(texts, self.sev_encoder.transform(severity_labels))
        self.use_ml = True
        logger.info("ML models trained and ready.")

    def predict_ml(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.use_ml or self.loss_pipeline is None or self.sev_pipeline is None:
            return None, None
        loss_idx = self.loss_pipeline.predict([text])[0]
        sev_idx = self.sev_pipeline.predict([text])[0]
        loss_label = self.loss_encoder.inverse_transform([loss_idx])[0]
        sev_label = self.sev_encoder.inverse_transform([sev_idx])[0]
        return loss_label, sev_label

    # ---------- Main normalize ----------
    def normalize(self, claim_text: str) -> NormalizedClaim:
        claim_raw = claim_text.strip()
        entities = self._extract_entities(claim_raw)
        # Rule-based predictions
        loss_types_rule = self._detect_loss_types_rule(claim_raw)
        severity_rule = self._determine_severity_rule(claim_raw, entities)
        assets = self._extract_assets(claim_raw)

        # ML predictions optionally
        pred_loss = None
        pred_sev = None
        if self.use_ml:
            ml_loss, ml_sev = self.predict_ml(claim_raw)
            pred_loss = ml_loss
            pred_sev = ml_sev

        # choose final loss types (prefer ML if available)
        final_loss = [lt.value for lt in loss_types_rule]
        final_severity = severity_rule.value

        if pred_loss:
            # ML returns single label; keep it as primary
            final_loss = [pred_loss]
        if pred_sev:
            final_severity = pred_sev

        conf = self._calculate_confidence(loss_types_rule, severity_rule, assets, entities)

        return NormalizedClaim(
            loss_types=final_loss,
            severity=final_severity,
            affected_assets=assets,
            confidence_score=conf,
            extracted_entities=entities,
            raw_text=claim_raw,
            predicted_loss=pred_loss,
            predicted_severity=pred_sev
        )

    # ---------- Evaluation helper ----------
    def evaluate(self, texts: List[str], true_loss: List[str], true_sev: List[str]) -> Dict[str, Any]:
        """
        Produce evaluation metrics for current configuration (rule-based or ML-based).
        Returns dict with accuracy/f1 for both outputs.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        preds_loss = []
        preds_sev = []
        for t in texts:
            r = self.normalize(t)
            # if ML predicted, r.predicted_loss will have a string else use rule-based
            preds_loss.append(r.predicted_loss if r.predicted_loss else (r.loss_types[0] if r.loss_types else "Other"))
            preds_sev.append(r.predicted_severity if r.predicted_severity else r.severity)

        metrics = {
            "loss": {
                "accuracy": float(accuracy_score(true_loss, preds_loss)),
                "f1_macro": float(f1_score(true_loss, preds_loss, average="macro", zero_division=0)),
                "precision_macro": float(precision_score(true_loss, preds_loss, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(true_loss, preds_loss, average="macro", zero_division=0)),
            },
            "severity": {
                "accuracy": float(accuracy_score(true_sev, preds_sev)),
                "f1_macro": float(f1_score(true_sev, preds_sev, average="macro", zero_division=0)),
                "precision_macro": float(precision_score(true_sev, preds_sev, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(true_sev, preds_sev, average="macro", zero_division=0)),
            }
        }
        return metrics

    def to_json(self, normalized: NormalizedClaim) -> str:
        return json.dumps(normalized.to_dict(), indent=2)

# End of file
