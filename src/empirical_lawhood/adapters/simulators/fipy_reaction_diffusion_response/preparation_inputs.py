"""Reviewed full numerical seed commitments for retained native preparation units."""

from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponsePhase
from empirical_lawhood.adapters.methods.selective_dependence_response.preparation_inputs import SelectiveDependenceResponsePreparationInput
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import FipyReactionDiffusionResponseFiPyDesign


DRAW_VARIABLE_IDS = ('background-amplitude', 'residual-amplitude', 'residual-centre', 'residual-width')
_COMPLETE_UNIT_NUMERICAL_INPUTS = {
    'fipy.selective-dependence-response.canary.unit-000': (91700598462294935799189995171146618875, 'CANARY', 0),
    'fipy.selective-dependence-response.canary.unit-001': (29460382875014753697260775678010204944, 'CANARY', 1),
    'fipy.selective-dependence-response.canary.unit-002': (75907899779460208976887957374076592081, 'CANARY', 2),
    'fipy.selective-dependence-response.canary.unit-003': (273658987186879253274825471186598084899, 'CANARY', 3),
    'fipy.selective-dependence-response.development.unit-000': (319691994064051881851123647262833409168, 'DEVELOPMENT', 0),
    'fipy.selective-dependence-response.development.unit-001': (144026514842444588000077649780272063595, 'DEVELOPMENT', 1),
    'fipy.selective-dependence-response.development.unit-002': (108960203475793767879106751111582268342, 'DEVELOPMENT', 2),
    'fipy.selective-dependence-response.development.unit-003': (244568705851896787311128276161603587123, 'DEVELOPMENT', 3),
    'fipy.selective-dependence-response.development.unit-004': (219804745130913433781429295642181001842, 'DEVELOPMENT', 4),
    'fipy.selective-dependence-response.development.unit-005': (125686862817375776498185870532370978023, 'DEVELOPMENT', 5),
    'fipy.selective-dependence-response.development.unit-006': (50871791104625415727567248630961457209, 'DEVELOPMENT', 6),
    'fipy.selective-dependence-response.development.unit-007': (8646369331952666734648807536202880560, 'DEVELOPMENT', 7),
    'fipy.selective-dependence-response.development.unit-008': (138309081624393529154877409088065799361, 'DEVELOPMENT', 8),
    'fipy.selective-dependence-response.development.unit-009': (70729788777987058198290271598812267558, 'DEVELOPMENT', 9),
    'fipy.selective-dependence-response.development.unit-010': (223580872908807801223148671571117514866, 'DEVELOPMENT', 10),
    'fipy.selective-dependence-response.development.unit-011': (224934910136287010920748008785233795730, 'DEVELOPMENT', 11),
    'fipy.selective-dependence-response.development.unit-012': (15211695739304239383206760121143847993, 'DEVELOPMENT', 12),
    'fipy.selective-dependence-response.development.unit-013': (249484372421509207559456962841680899954, 'DEVELOPMENT', 13),
    'fipy.selective-dependence-response.development.unit-014': (147356391017750946218512823427631592972, 'DEVELOPMENT', 14),
    'fipy.selective-dependence-response.development.unit-015': (82405784352238248222688135664379301196, 'DEVELOPMENT', 15),
    'fipy.selective-dependence-response.development.unit-016': (126886900729758918082458104665478546040, 'DEVELOPMENT', 16),
    'fipy.selective-dependence-response.development.unit-017': (118817033109336094384666671854013635471, 'DEVELOPMENT', 17),
    'fipy.selective-dependence-response.development.unit-018': (83951897206246524827969836039459374569, 'DEVELOPMENT', 18),
    'fipy.selective-dependence-response.development.unit-019': (231342665022299892778371892415804547047, 'DEVELOPMENT', 19),
    'fipy.selective-dependence-response.development.unit-020': (333995254988736999887714379434602219450, 'DEVELOPMENT', 20),
    'fipy.selective-dependence-response.development.unit-021': (330039651619669588393426292918895845811, 'DEVELOPMENT', 21),
    'fipy.selective-dependence-response.development.unit-022': (240445055779316041653247859914597188310, 'DEVELOPMENT', 22),
    'fipy.selective-dependence-response.development.unit-023': (24377469680948500086701207996614715412, 'DEVELOPMENT', 23),
    'fipy.selective-dependence-response.development.unit-024': (278665773286275935720120331906880741306, 'DEVELOPMENT', 24),
    'fipy.selective-dependence-response.development.unit-025': (129077954582225678346767885189133556382, 'DEVELOPMENT', 25),
    'fipy.selective-dependence-response.development.unit-026': (312733043397435947975785896083343313146, 'DEVELOPMENT', 26),
    'fipy.selective-dependence-response.development.unit-027': (331855480417915379769031023116470030684, 'DEVELOPMENT', 27),
    'fipy.selective-dependence-response.development.unit-028': (320073900956234145682637838958360917632, 'DEVELOPMENT', 28),
    'fipy.selective-dependence-response.development.unit-029': (61826280532945491842142902146415670432, 'DEVELOPMENT', 29),
    'fipy.selective-dependence-response.development.unit-030': (25851158394854538700098976441904928711, 'DEVELOPMENT', 30),
    'fipy.selective-dependence-response.development.unit-031': (82206462152048671492006694644885465121, 'DEVELOPMENT', 31),
    'fipy.selective-dependence-response.evaluation.unit-000': (273843350488070473099776843398598103015, 'EVALUATION', 0),
    'fipy.selective-dependence-response.evaluation.unit-001': (246259567410550493012908244214966160124, 'EVALUATION', 1),
    'fipy.selective-dependence-response.evaluation.unit-002': (6200274838454328576691629018500056178, 'EVALUATION', 2),
    'fipy.selective-dependence-response.evaluation.unit-003': (125010428221575523980073648761131800734, 'EVALUATION', 3),
    'fipy.selective-dependence-response.evaluation.unit-004': (339632458285660073867218793246893444295, 'EVALUATION', 4),
    'fipy.selective-dependence-response.evaluation.unit-005': (244254050707799802943879082992375105983, 'EVALUATION', 5),
    'fipy.selective-dependence-response.evaluation.unit-006': (290124905488269204080842914929400230488, 'EVALUATION', 6),
    'fipy.selective-dependence-response.evaluation.unit-007': (277598117821968502054874122391079514394, 'EVALUATION', 7),
    'fipy.selective-dependence-response.evaluation.unit-008': (96923518394160657211317290435595583055, 'EVALUATION', 8),
    'fipy.selective-dependence-response.evaluation.unit-009': (150573789447950734583312688591513724996, 'EVALUATION', 9),
    'fipy.selective-dependence-response.evaluation.unit-010': (334332569655137027463123241436757973590, 'EVALUATION', 10),
    'fipy.selective-dependence-response.evaluation.unit-011': (155701513602270600935286057185617207138, 'EVALUATION', 11),
    'fipy.selective-dependence-response.evaluation.unit-012': (35556329520611323209170402197740898370, 'EVALUATION', 12),
    'fipy.selective-dependence-response.evaluation.unit-013': (294420946465780774987484107261755204141, 'EVALUATION', 13),
    'fipy.selective-dependence-response.evaluation.unit-014': (145756833507063392997751210612143812299, 'EVALUATION', 14),
    'fipy.selective-dependence-response.evaluation.unit-015': (119600412673928181128587974007433985289, 'EVALUATION', 15),
    'fipy.selective-dependence-response.evaluation.unit-016': (315428632101421195026956882543771203316, 'EVALUATION', 16),
    'fipy.selective-dependence-response.evaluation.unit-017': (182147365828272480651328636572509069302, 'EVALUATION', 17),
    'fipy.selective-dependence-response.evaluation.unit-018': (228652546734297468543257556586513072897, 'EVALUATION', 18),
    'fipy.selective-dependence-response.evaluation.unit-019': (101288746573110562358338322333275079361, 'EVALUATION', 19),
    'fipy.selective-dependence-response.evaluation.unit-020': (320413190884283466372948686605700243586, 'EVALUATION', 20),
    'fipy.selective-dependence-response.evaluation.unit-021': (200241055392598321019873484736534702825, 'EVALUATION', 21),
    'fipy.selective-dependence-response.evaluation.unit-022': (50662606542243381481397526395369830324, 'EVALUATION', 22),
    'fipy.selective-dependence-response.evaluation.unit-023': (53563733111825395657091837825774633516, 'EVALUATION', 23),
    'fipy.selective-dependence-response.evaluation.unit-024': (319117179999908210037804465969809390537, 'EVALUATION', 24),
    'fipy.selective-dependence-response.evaluation.unit-025': (314427317435811061191866346179054115051, 'EVALUATION', 25),
    'fipy.selective-dependence-response.evaluation.unit-026': (140005278180117791625704147789774038486, 'EVALUATION', 26),
    'fipy.selective-dependence-response.evaluation.unit-027': (313166989529121255963300512609423008177, 'EVALUATION', 27),
    'fipy.selective-dependence-response.evaluation.unit-028': (319678444756679774328203634159408647203, 'EVALUATION', 28),
    'fipy.selective-dependence-response.evaluation.unit-029': (278422648509608627195905607259409204285, 'EVALUATION', 29),
    'fipy.selective-dependence-response.evaluation.unit-030': (225065239057020129592379109045992204101, 'EVALUATION', 30),
    'fipy.selective-dependence-response.evaluation.unit-031': (70721085449042180693582920853056884179, 'EVALUATION', 31),
    'fipy.selective-dependence-response.evaluation.unit-032': (295966775106661828612046800388342900620, 'EVALUATION', 32),
    'fipy.selective-dependence-response.evaluation.unit-033': (85440763798803243150579118035491096592, 'EVALUATION', 33),
    'fipy.selective-dependence-response.evaluation.unit-034': (10573294594029494036423754513439028478, 'EVALUATION', 34),
    'fipy.selective-dependence-response.evaluation.unit-035': (184886010922364747714086348831969250803, 'EVALUATION', 35),
    'fipy.selective-dependence-response.evaluation.unit-036': (258223974525266691021801650392456598762, 'EVALUATION', 36),
    'fipy.selective-dependence-response.evaluation.unit-037': (302951444511783475245468970458243340195, 'EVALUATION', 37),
    'fipy.selective-dependence-response.evaluation.unit-038': (225671462484119609677296237640295374122, 'EVALUATION', 38),
    'fipy.selective-dependence-response.evaluation.unit-039': (335739564637669401804262067225998176451, 'EVALUATION', 39),
    'fipy.selective-dependence-response.evaluation.unit-040': (300008985377673120303153106879604802229, 'EVALUATION', 40),
    'fipy.selective-dependence-response.evaluation.unit-041': (134351128376505374944113349472394284444, 'EVALUATION', 41),
    'fipy.selective-dependence-response.evaluation.unit-042': (329835147380866328149723032757592698657, 'EVALUATION', 42),
    'fipy.selective-dependence-response.evaluation.unit-043': (20594784874190218884388271722778637751, 'EVALUATION', 43),
    'fipy.selective-dependence-response.evaluation.unit-044': (42575016292343719867289065582108898805, 'EVALUATION', 44),
    'fipy.selective-dependence-response.evaluation.unit-045': (198205322209816641985186082922493193502, 'EVALUATION', 45),
    'fipy.selective-dependence-response.evaluation.unit-046': (174182537673915128150809793329806727306, 'EVALUATION', 46),
    'fipy.selective-dependence-response.evaluation.unit-047': (86926200497422090166656858490380820149, 'EVALUATION', 47),
    'fipy.selective-dependence-response.evaluation.unit-048': (125299730796140568741635317650691675978, 'EVALUATION', 48),
    'fipy.selective-dependence-response.evaluation.unit-049': (313645612689598877426802534878031265985, 'EVALUATION', 49),
    'fipy.selective-dependence-response.evaluation.unit-050': (70517983154486785102755029929635397238, 'EVALUATION', 50),
    'fipy.selective-dependence-response.evaluation.unit-051': (175637291329349960366911336592949376325, 'EVALUATION', 51),
    'fipy.selective-dependence-response.evaluation.unit-052': (78542799159911049816965159946503938164, 'EVALUATION', 52),
    'fipy.selective-dependence-response.evaluation.unit-053': (219888739882718698104983538945089297193, 'EVALUATION', 53),
    'fipy.selective-dependence-response.evaluation.unit-054': (24228487465135693754746464170410500944, 'EVALUATION', 54),
    'fipy.selective-dependence-response.evaluation.unit-055': (218261945175032022861270084899405554308, 'EVALUATION', 55),
    'fipy.selective-dependence-response.evaluation.unit-056': (289906289021158095996836060343216829071, 'EVALUATION', 56),
    'fipy.selective-dependence-response.evaluation.unit-057': (137559589604023828792611206521732000548, 'EVALUATION', 57),
    'fipy.selective-dependence-response.evaluation.unit-058': (112029309975238829671930983098846568855, 'EVALUATION', 58),
    'fipy.selective-dependence-response.evaluation.unit-059': (169148605230593702205744186885617744907, 'EVALUATION', 59),
    'fipy.selective-dependence-response.evaluation.unit-060': (332717586200388139304791818241730343840, 'EVALUATION', 60),
    'fipy.selective-dependence-response.evaluation.unit-061': (307225566492717788617224256390649109653, 'EVALUATION', 61),
    'fipy.selective-dependence-response.evaluation.unit-062': (7289734168904579769597100656994218905, 'EVALUATION', 62),
    'fipy.selective-dependence-response.evaluation.unit-063': (139993431799833626087323598639377796108, 'EVALUATION', 63),
    'fipy.selective-dependence-response.reserve.unit-000': (308966994840355826347341554721082420568, 'RESERVE', 0),
    'fipy.selective-dependence-response.reserve.unit-001': (339130868255599378288603197303309175716, 'RESERVE', 1),
    'fipy.selective-dependence-response.reserve.unit-002': (42052263260115951399893796321307906570, 'RESERVE', 2),
    'fipy.selective-dependence-response.reserve.unit-003': (66277803436009813045509831795831658558, 'RESERVE', 3),
    'fipy.selective-dependence-response.reserve.unit-004': (91560112393940705456038256710790445366, 'RESERVE', 4),
    'fipy.selective-dependence-response.reserve.unit-005': (337370736847453688048695347157901250987, 'RESERVE', 5),
    'fipy.selective-dependence-response.reserve.unit-006': (89602339270976494688291978636736009888, 'RESERVE', 6),
    'fipy.selective-dependence-response.reserve.unit-007': (114671503362248626244224773579779982371, 'RESERVE', 7),
}


def fipy_preparation_input(
    design: FipyReactionDiffusionResponseFiPyDesign,
    complete_unit_id: str,
    phase: SelectiveDependenceResponsePhase,
) -> SelectiveDependenceResponsePreparationInput:
    if complete_unit_id not in _COMPLETE_UNIT_NUMERICAL_INPUTS:
        raise ValueError("native unit requires an explicit full numerical preparation input; no name-derived sampler is available")
    full_seed, phase_name, roster_index = _COMPLETE_UNIT_NUMERICAL_INPUTS[complete_unit_id]
    if phase.value != phase_name:
        raise ValueError("retained numerical preparation unit differs from the exact original phase")
    return SelectiveDependenceResponsePreparationInput(
        complete_unit_id=complete_unit_id,
        target_id=design.target_id,
        target_design=ObjectIdentity.from_record(design.design_id, design),
        phase=phase.value,
        roster_index=roster_index,
        full_seed=full_seed,
        draw_variable_ids=DRAW_VARIABLE_IDS,
    )


def validate_reviewed_preparation_input(value: SelectiveDependenceResponsePreparationInput) -> None:
    if value.complete_unit_id in _COMPLETE_UNIT_NUMERICAL_INPUTS:
        if (value.full_seed, value.phase, value.roster_index) != _COMPLETE_UNIT_NUMERICAL_INPUTS[value.complete_unit_id]:
            raise ValueError("retained preparation differs from the reviewed full seed/phase/roster position")
