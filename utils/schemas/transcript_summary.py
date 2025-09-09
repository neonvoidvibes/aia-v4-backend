from pydantic import BaseModel, Field, conint, confloat
from typing import List, Optional, Literal, Dict

# ----- Layer 1 -----
class MeetingMetadata(BaseModel):
    timestamp: str
    duration_minutes: conint(ge=0)
    participants_count: conint(ge=1)
    meeting_type: str
    primary_purpose: str

class Decision(BaseModel):
    decision: str
    context: str
    urgency: Literal["low", "medium", "high"]
    source_span: Optional[str] = None

class TaskItem(BaseModel):
    task: str
    owner_role: str
    deadline: Optional[str] = None  # YYYY-MM-DD
    dependencies: List[str] = []
    source_span: Optional[str] = None

class NextMeeting(BaseModel):
    purpose: str
    timeframe: str
    required_participants: List[str] = []
    source_span: Optional[str] = None

class ActionableOutputs(BaseModel):
    decisions: List[Decision] = []
    tasks: List[TaskItem] = []
    next_meetings: List[NextMeeting] = []

class Layer1(BaseModel):
    meeting_metadata: MeetingMetadata
    actionable_outputs: ActionableOutputs

# ----- Layer 2 -----
class ExplicitTheme(BaseModel):
    theme: str
    frequency: conint(ge=0)
    context: str
    segment_ids: List[str] = []

class StatedAgreement(BaseModel):
    agreement: str
    confidence: Literal["low", "medium", "high"]
    segment_ids: List[str] = []

class DirectQuote(BaseModel):
    quote: str
    context: str
    significance: str
    segment_ids: List[str] = []

class ParticipationPatterns(BaseModel):
    energy_shifts: List[str] = []
    engagement_distribution: str

class MirrorLevel(BaseModel):
    explicit_themes: List[ExplicitTheme] = []
    stated_agreements: List[StatedAgreement] = []
    direct_quotes: List[DirectQuote] = []
    participation_patterns: ParticipationPatterns

class HiddenPattern(BaseModel):
    pattern: str
    evidence: List[str] = []            # pointers into mirror
    systemic_significance: str
    evidence_ids: List[str] = []        # normalized refs

class UnspokenTension(BaseModel):
    tension: str
    indicators: List[str] = []
    impact_assessment: str
    evidence_ids: List[str] = []

class GroupDynamics(BaseModel):
    emotional_undercurrents: str
    power_dynamics: str

class Paradox(BaseModel):
    paradox: str
    implications: str
    evidence_ids: List[str] = []

class LensLevel(BaseModel):
    hidden_patterns: List[HiddenPattern] = []
    unspoken_tensions: List[UnspokenTension] = []
    group_dynamics: GroupDynamics
    paradoxes_contradictions: List[Paradox] = []

class EmergentPossibility(BaseModel):
    possibility: str
    transformation_potential: str
    grounding_in_lens: str
    evidence_ids: List[str] = []

class InterventionOpportunity(BaseModel):
    leverage_point: str
    predicted_outcomes: List[str] = []
    probability_score: confloat(ge=0.0, le=1.0)
    evidence_ids: List[str] = []

class ParadigmShift(BaseModel):
    shift: str
    indicators: str
    readiness_assessment: str
    evidence_ids: List[str] = []

class PortalLevel(BaseModel):
    emergent_possibilities: List[EmergentPossibility] = []
    intervention_opportunities: List[InterventionOpportunity] = []
    paradigm_shifts: List[ParadigmShift] = []

class Layer2(BaseModel):
    mirror_level: MirrorLevel
    lens_level: LensLevel
    portal_level: PortalLevel

# ----- Layer 3 -----
class SAItem(BaseModel):
    insight: str
    individual_growth: str

class IDItem(BaseModel):
    relationship_shift: str
    collaboration_quality: str

class SUItem(BaseModel):
    systems_insight: str
    broader_context: str

class ConnectednessPatterns(BaseModel):
    self_awareness: List[SAItem] = []
    interpersonal_dynamics: List[IDItem] = []
    systemic_understanding: List[SUItem] = []

class BeautyItem(BaseModel):
    aesthetic_insight: str
    elegance_factor: str

class TruthItem(BaseModel):
    truth_revealed: str
    reality_alignment: str

class GoodnessItem(BaseModel):
    life_affirming_direction: str
    stakeholder_benefit: str

class TranscendentalAlignment(BaseModel):
    beauty_moments: List[BeautyItem] = []
    truth_emergence: List[TruthItem] = []
    goodness_orientation: List[GoodnessItem] = []
    coherence_quality: str

class SentienceItem(BaseModel):
    awareness_deepening: str
    empathy_development: str

class IntelligenceItem(BaseModel):
    sense_making_advancement: str
    complexity_navigation: str

class AgencyItem(BaseModel):
    responsibility_taking: str
    purposeful_action: str

class SovereigntyDevelopment(BaseModel):
    sentience_expansion: List[SentienceItem] = []
    intelligence_integration: List[IntelligenceItem] = []
    agency_manifestation: List[AgencyItem] = []

class Layer3(BaseModel):
    connectedness_patterns: ConnectednessPatterns
    transcendental_alignment: TranscendentalAlignment
    sovereignty_development: SovereigntyDevelopment

# ----- Layer 4 -----
class SingleLoopItem(BaseModel):
    error_correction: str
    process_improvement: str

class DoubleLoopItem(BaseModel):
    assumption_questioning: str
    mental_model_shift: str

class TripleLoopItem(BaseModel):
    context_examination: str
    paradigm_transformation: str

class TripleLoopLearning(BaseModel):
    single_loop: List[SingleLoopItem] = []
    double_loop: List[DoubleLoopItem] = []
    triple_loop: List[TripleLoopItem] = []

class RelationalInsight(BaseModel):
    relationship_between: List[str]
    pattern: str
    systemic_impact: str

class TranscontextualConnection(BaseModel):
    contexts: List[str]
    emergent_property: str

class LivingSystemsRecognition(BaseModel):
    system_characteristic: str
    health_indicator: str

class WarmDataPatterns(BaseModel):
    relational_insights: List[RelationalInsight] = []
    transcontextual_connections: List[TranscontextualConnection] = []
    living_systems_recognition: List[LivingSystemsRecognition] = []

class InsightCaptured(BaseModel):
    insight: str
    application_potential: str
    integration_path: str

class WisdomMoment(BaseModel):
    wisdom_expression: str
    depth_indicator: str
    collective_impact: str

class CapacityBuilding(BaseModel):
    capacity: str
    development_trajectory: str

class KnowledgeEvolution(BaseModel):
    insights_captured: List[InsightCaptured] = []
    wisdom_moments: List[WisdomMoment] = []
    capacity_building: List[CapacityBuilding] = []

class Layer4(BaseModel):
    triple_loop_learning: TripleLoopLearning
    warm_data_patterns: WarmDataPatterns
    knowledge_evolution: KnowledgeEvolution

# ----- Envelope -----
class FullSummary(BaseModel):
    layer1: Layer1
    layer2: Layer2
    layer3: Layer3
    layer4: Layer4
    confidence: Dict[str, float] = Field(default_factory=dict)

