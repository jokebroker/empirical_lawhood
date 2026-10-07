"""Reviewed full numerical seed commitments for retained native preparation units."""

from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponsePhase
from empirical_lawhood.adapters.methods.selective_dependence_response.preparation_inputs import SelectiveDependenceResponsePreparationInput
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import CanteraReactionResponseCanteraDesign


DRAW_VARIABLE_IDS = ('bath-temperature', 'cold-checkpoint-temperature', 'equivalence-ratio', 'heat-transfer-scale', 'hot-checkpoint-seed-temperature', 'inlet-temperature', 'reactor-volume')
_COMPLETE_UNIT_NUMERICAL_INPUTS = {
    'cantera.selective-dependence-response.canary.unit-000': (267051667907994525814370925456516897772, 'CANARY', 0),
    'cantera.selective-dependence-response.canary.unit-001': (164931752584159812297606576391683710778, 'CANARY', 1),
    'cantera.selective-dependence-response.canary.unit-002': (270836535681261289249066727105242610720, 'CANARY', 2),
    'cantera.selective-dependence-response.canary.unit-003': (338439877216051762338371992945763781576, 'CANARY', 3),
    'cantera.selective-dependence-response.development.unit-000': (23373246591605114562393651043966633070, 'DEVELOPMENT', 0),
    'cantera.selective-dependence-response.development.unit-001': (295865205771931898258910916921804479693, 'DEVELOPMENT', 1),
    'cantera.selective-dependence-response.development.unit-002': (235270825500509994552949672024768587432, 'DEVELOPMENT', 2),
    'cantera.selective-dependence-response.development.unit-003': (224531705797034822439265098837612119763, 'DEVELOPMENT', 3),
    'cantera.selective-dependence-response.development.unit-004': (166387110911764908878565723870965252370, 'DEVELOPMENT', 4),
    'cantera.selective-dependence-response.development.unit-005': (193678833857905319655952273180675682454, 'DEVELOPMENT', 5),
    'cantera.selective-dependence-response.development.unit-006': (225852250698447802870003288588247377831, 'DEVELOPMENT', 6),
    'cantera.selective-dependence-response.development.unit-007': (149591559116840267585954480953782001343, 'DEVELOPMENT', 7),
    'cantera.selective-dependence-response.development.unit-008': (111861681724046513716596512677859781035, 'DEVELOPMENT', 8),
    'cantera.selective-dependence-response.development.unit-009': (126599603585762542702976549700299791462, 'DEVELOPMENT', 9),
    'cantera.selective-dependence-response.development.unit-010': (75241663154039107890213982473997002418, 'DEVELOPMENT', 10),
    'cantera.selective-dependence-response.development.unit-011': (240442918521858174529019716376112628610, 'DEVELOPMENT', 11),
    'cantera.selective-dependence-response.development.unit-012': (289797876902906011865671247371337244450, 'DEVELOPMENT', 12),
    'cantera.selective-dependence-response.development.unit-013': (337402335172345722523351817739517119564, 'DEVELOPMENT', 13),
    'cantera.selective-dependence-response.development.unit-014': (253707565164280446189714980574130440285, 'DEVELOPMENT', 14),
    'cantera.selective-dependence-response.development.unit-015': (157112934791906256468844978087656945046, 'DEVELOPMENT', 15),
    'cantera.selective-dependence-response.development.unit-016': (188274939668107753091505054998773625235, 'DEVELOPMENT', 16),
    'cantera.selective-dependence-response.development.unit-017': (140430031417634216085349570341652003268, 'DEVELOPMENT', 17),
    'cantera.selective-dependence-response.development.unit-018': (45468093346278225189451093069834856089, 'DEVELOPMENT', 18),
    'cantera.selective-dependence-response.development.unit-019': (148548264757504520541097203361965202933, 'DEVELOPMENT', 19),
    'cantera.selective-dependence-response.development.unit-020': (138325378543802594995075663682359263075, 'DEVELOPMENT', 20),
    'cantera.selective-dependence-response.development.unit-021': (136518283346908801364032581333617087361, 'DEVELOPMENT', 21),
    'cantera.selective-dependence-response.development.unit-022': (307330328743411503531782485492788121251, 'DEVELOPMENT', 22),
    'cantera.selective-dependence-response.development.unit-023': (330941098728797322458210596523195383573, 'DEVELOPMENT', 23),
    'cantera.selective-dependence-response.evaluation.unit-000': (29087249295511247062078833982651688236, 'EVALUATION', 0),
    'cantera.selective-dependence-response.evaluation.unit-001': (208946718491265996666021310225232652434, 'EVALUATION', 1),
    'cantera.selective-dependence-response.evaluation.unit-002': (220343793918208732579791625071970932063, 'EVALUATION', 2),
    'cantera.selective-dependence-response.evaluation.unit-003': (84974159742232536630825801018856584426, 'EVALUATION', 3),
    'cantera.selective-dependence-response.evaluation.unit-004': (186434575205135288110295260836408280444, 'EVALUATION', 4),
    'cantera.selective-dependence-response.evaluation.unit-005': (189979636164788371883421063160865909665, 'EVALUATION', 5),
    'cantera.selective-dependence-response.evaluation.unit-006': (168539058738772520887785173073082696734, 'EVALUATION', 6),
    'cantera.selective-dependence-response.evaluation.unit-007': (248458267711267208499160982087319688730, 'EVALUATION', 7),
    'cantera.selective-dependence-response.evaluation.unit-008': (129125249894054641855095223847906962429, 'EVALUATION', 8),
    'cantera.selective-dependence-response.evaluation.unit-009': (324246797084476000517952885413945686371, 'EVALUATION', 9),
    'cantera.selective-dependence-response.evaluation.unit-010': (70049861794996989428917688950669524276, 'EVALUATION', 10),
    'cantera.selective-dependence-response.evaluation.unit-011': (201527585744149199249444723065911793587, 'EVALUATION', 11),
    'cantera.selective-dependence-response.evaluation.unit-012': (318520788103075270087791577347644586439, 'EVALUATION', 12),
    'cantera.selective-dependence-response.evaluation.unit-013': (269377689277210460588180769124324077515, 'EVALUATION', 13),
    'cantera.selective-dependence-response.evaluation.unit-014': (162392739825550505330170958999597161765, 'EVALUATION', 14),
    'cantera.selective-dependence-response.evaluation.unit-015': (184032285840126974562188012564859439173, 'EVALUATION', 15),
    'cantera.selective-dependence-response.evaluation.unit-016': (723605816334107944240643283554601048, 'EVALUATION', 16),
    'cantera.selective-dependence-response.evaluation.unit-017': (223889130918122577422221503115555635538, 'EVALUATION', 17),
    'cantera.selective-dependence-response.evaluation.unit-018': (184037127138592401741839702836411069932, 'EVALUATION', 18),
    'cantera.selective-dependence-response.evaluation.unit-019': (152246313834828143265485105512413126565, 'EVALUATION', 19),
    'cantera.selective-dependence-response.evaluation.unit-020': (219185354996378615400578557761855281624, 'EVALUATION', 20),
    'cantera.selective-dependence-response.evaluation.unit-021': (302289924681623850067058305369905554535, 'EVALUATION', 21),
    'cantera.selective-dependence-response.evaluation.unit-022': (74668083193196926774204931169628953492, 'EVALUATION', 22),
    'cantera.selective-dependence-response.evaluation.unit-023': (324745704472828190460227050038443972689, 'EVALUATION', 23),
    'cantera.selective-dependence-response.evaluation.unit-024': (191765402479786362829765184896483548986, 'EVALUATION', 24),
    'cantera.selective-dependence-response.evaluation.unit-025': (112835704151028555804014263023883516340, 'EVALUATION', 25),
    'cantera.selective-dependence-response.evaluation.unit-026': (72764889209001737758794095738236984804, 'EVALUATION', 26),
    'cantera.selective-dependence-response.evaluation.unit-027': (53687295176606022794895080034047571535, 'EVALUATION', 27),
    'cantera.selective-dependence-response.evaluation.unit-028': (312566640873612357432287344073619356139, 'EVALUATION', 28),
    'cantera.selective-dependence-response.evaluation.unit-029': (8600002452644824548865864639594233229, 'EVALUATION', 29),
    'cantera.selective-dependence-response.evaluation.unit-030': (77170248860104429996587151194014654628, 'EVALUATION', 30),
    'cantera.selective-dependence-response.evaluation.unit-031': (57021069677392364818940009188485084579, 'EVALUATION', 31),
    'cantera.selective-dependence-response.evaluation.unit-032': (84960516281323433069475631577370074725, 'EVALUATION', 32),
    'cantera.selective-dependence-response.evaluation.unit-033': (154831024824386752424825027615497658567, 'EVALUATION', 33),
    'cantera.selective-dependence-response.evaluation.unit-034': (129586478869365167100816031449763824638, 'EVALUATION', 34),
    'cantera.selective-dependence-response.evaluation.unit-035': (317510786310669801461108952225873534630, 'EVALUATION', 35),
    'cantera.selective-dependence-response.evaluation.unit-036': (316935935105024556674713500407379094606, 'EVALUATION', 36),
    'cantera.selective-dependence-response.evaluation.unit-037': (84164321480554242972637697471395419691, 'EVALUATION', 37),
    'cantera.selective-dependence-response.evaluation.unit-038': (263644758587454198537525082368037411457, 'EVALUATION', 38),
    'cantera.selective-dependence-response.evaluation.unit-039': (224804715237827515083839743420286838008, 'EVALUATION', 39),
    'cantera.selective-dependence-response.evaluation.unit-040': (229769546656347476916063668749780445790, 'EVALUATION', 40),
    'cantera.selective-dependence-response.evaluation.unit-041': (238310083688008866439098599170673303891, 'EVALUATION', 41),
    'cantera.selective-dependence-response.evaluation.unit-042': (300832961590092668243342607769993420672, 'EVALUATION', 42),
    'cantera.selective-dependence-response.evaluation.unit-043': (188625465036599474959147247648829404781, 'EVALUATION', 43),
    'cantera.selective-dependence-response.evaluation.unit-044': (16295743053908678730957578553471901405, 'EVALUATION', 44),
    'cantera.selective-dependence-response.evaluation.unit-045': (334951065321396677970119143735038779460, 'EVALUATION', 45),
    'cantera.selective-dependence-response.evaluation.unit-046': (98320597790904932984923728931586243797, 'EVALUATION', 46),
    'cantera.selective-dependence-response.evaluation.unit-047': (41258049814089367226275807116283050379, 'EVALUATION', 47),
    'cantera.selective-dependence-response.evaluation.unit-048': (70527935245203877674357503319307757136, 'EVALUATION', 48),
    'cantera.selective-dependence-response.evaluation.unit-049': (312396851694454486638038411613712131560, 'EVALUATION', 49),
    'cantera.selective-dependence-response.evaluation.unit-050': (197676289823624372179645327764011164661, 'EVALUATION', 50),
    'cantera.selective-dependence-response.evaluation.unit-051': (275507162431381924385179880659678503802, 'EVALUATION', 51),
    'cantera.selective-dependence-response.evaluation.unit-052': (337767696869116379666141124379595189245, 'EVALUATION', 52),
    'cantera.selective-dependence-response.evaluation.unit-053': (261885890663843074468654152708076498524, 'EVALUATION', 53),
    'cantera.selective-dependence-response.evaluation.unit-054': (115076387872468846136723193956753110817, 'EVALUATION', 54),
    'cantera.selective-dependence-response.evaluation.unit-055': (129839450249769152690756068212114925339, 'EVALUATION', 55),
    'cantera.selective-dependence-response.evaluation.unit-056': (24165578802960380642498561252869322387, 'EVALUATION', 56),
    'cantera.selective-dependence-response.evaluation.unit-057': (322593420396120152744493298518231591127, 'EVALUATION', 57),
    'cantera.selective-dependence-response.evaluation.unit-058': (105106602216156200559173431867268039562, 'EVALUATION', 58),
    'cantera.selective-dependence-response.evaluation.unit-059': (196778490788695235146895600708597393102, 'EVALUATION', 59),
    'cantera.selective-dependence-response.evaluation.unit-060': (174023073955459693883672493091347483683, 'EVALUATION', 60),
    'cantera.selective-dependence-response.evaluation.unit-061': (43273268356597964960243299008075049045, 'EVALUATION', 61),
    'cantera.selective-dependence-response.evaluation.unit-062': (101761945154346850290944083933704596045, 'EVALUATION', 62),
    'cantera.selective-dependence-response.evaluation.unit-063': (96063102396271318732746208082608964473, 'EVALUATION', 63),
    'cantera.selective-dependence-response.reserve.unit-000': (255198112063958510354732502227676676724, 'RESERVE', 0),
    'cantera.selective-dependence-response.reserve.unit-001': (98671427799892043162342492136464246302, 'RESERVE', 1),
    'cantera.selective-dependence-response.reserve.unit-002': (310536170672119919216436412300335886763, 'RESERVE', 2),
    'cantera.selective-dependence-response.reserve.unit-003': (299067491167032959675672316300198943201, 'RESERVE', 3),
    'cantera.selective-dependence-response.reserve.unit-004': (276767297276178507696750062103103893155, 'RESERVE', 4),
    'cantera.selective-dependence-response.reserve.unit-005': (211402171523992560803875801862891148144, 'RESERVE', 5),
    'cantera.selective-dependence-response.reserve.unit-006': (106766013001236096065820439441433465854, 'RESERVE', 6),
    'cantera.selective-dependence-response.reserve.unit-007': (164556332868004565917584505142705095364, 'RESERVE', 7),
}


def cantera_preparation_input(
    design: CanteraReactionResponseCanteraDesign,
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
