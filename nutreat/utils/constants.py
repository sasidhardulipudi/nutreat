from pathlib import Path

# Home folder or Root for the project.
PROJECT_HOME = str(Path().absolute()).rsplit('/', 1)[0]

# Nutrient details file contains all 7000+ food items along with all
# the nutrient details.
NUTRIENT_DETAILS_FILE = PROJECT_HOME + '/data/nutrient_details.txt'

# Theta file is used to checkpoint computed theta value
PREVIOUS_THETA_FILE = PROJECT_HOME + "/data/theta.txt"

# Cost file is use to checkpoint computed cost value
PREVIOUS_COST_FILE = PROJECT_HOME + "/data/previous_cost.txt"

# Input file is used to check point previous input.
INPUT_FILE = PROJECT_HOME + "/data/input.txt"

# Output file is used to display the output in a table format.
OUTPUT_FILE = PROJECT_HOME + '/data/output.txt'

#
DAILY_NUTRIENTS_LIMIT_FILE = PROJECT_HOME + '/data/daily_nutrients_limit.txt'
NORMALISED_OUTPUT_FILE = PROJECT_HOME + '/data/normalised_output.txt'
TOP_ALL_FOODS_PER_NUTRIENT_FILE = \
    PROJECT_HOME \
    + '/data/top_all_foods_per_nutrient.pickle'
TOP_SELECTED_FOODS_PER_NUTRIENT_FILE = \
    PROJECT_HOME \
    + '/data/top_selected_foods_per_nutrient.pickle'
NUTRIENT_START_INDEX = 15
NUTRIENT_END_INDEX = 163
GRAMS = 100
SAMPLE_SIZE = 1
ALPHA = 0.000000010090098009999999999999999999999999999999
# BETA1 = 0.9
# BETA2 = 0.999
BETA1 = -99
BETA2 = -99
EPSILON = 10 ** (-8)
FIRST_RUN = True
ITERATIONS = 1000
MAX_COST = 999999999999999999999999
SAVE_FREQUENCY = 99
NUTRIENT_LIST = [
    '10:0_carpic_acid_g', '12:0_Lauric_Acid_g', '13:0_Tridecanoic_acid_g',
    '14:0_Miristic_Acid_g', '14:1_Miristoleic_acid_g',
    '15:0_Pentadecanoic_acid_g', '15:1_g', '16:0_g', '16:1_c_g', '16:1_t_g',
    '16:1_undifferentiated_g', '17:0_g', '17:1_g', '18:0_g',
    '18:1_11_t_18:1t_n_7_g', '18:1_c_g', '18:1_t_g',
    '18:1_undifferentiated_g', '18:2_CLAs_g', '18:2_i_g',
    '18:2_n_6_c_Linoleic_acid_omega_6c_g', '18:2_t_not_further_defined_g',
    '18:2_t_t_g', '18:2_undifferentiated_g',
    '18:3_n_3_ccc_alpha_Linolenic_acid_omega_3_ALA_g',
    '18:3_n_6_ccc_Gamma_linolenic_acid_omega_6_g', '18:3_undifferentiated_g',
    '18:3i_g', '18:4_g', '20:0_g', '20:1_g', '20:2_n_6_cc_g', '20:3_n_3_g',
    '20:3_n_6_g', '20:3_undifferentiated_g', '20:4_n_6_g',
    '20:4_undifferentiated_g', '20:5_n_3_EPAeicosapentaenoic_acid_omega_3_g',
    '21:5_g', '22:0_g', '22:1_c_g', '22:1_t_g', '22:1_undifferentiated_g',
    '22:4_g', '22:5_n_3_DPA_g', '22:6_n_3_DHA_docosahexaenoic_acid_omega_3_g',
    '24:0_g', '24:1_c_g', '4:0_g', '6:0_g', '8:0_g', 'Alanine_g',
    'Alcohol_ethyl_g', 'Arginine_g', 'Ash_g', 'Aspartic_acid_g',
    'Beta_sitosterol_mg', 'Betaine_mg', 'Caffeine_mg', 'Calcium_Ca_mg',
    'Campesterol_mg', 'Carbohydrate_by_difference_g', 'Carotene_alpha_mcg',
    'Carotene_beta_mcg', 'Cholesterol_mg', 'Choline_total_mg', 'Copper_Cu_mg',
    'Cryptoxanthin_beta_mcg', 'Cystine_g', 'Dihydrophylloquinone_mcg',
    'Energy_kJ', 'Energy_kcal', 'monounsaturated_Fatty_acids_total_g',
    'polyunsaturated_Fatty_acids_total_g', 'saturated_Fatty_acids_total_g',
    'trans_Fatty_acids_total_g', 'trans_monoenoic_Fatty_acids_total_g',
    'trans_polyenoic_Fatty_acids_total_g', 'Fiber_total_dietary_g',
    'Fluoride_F_mcg', 'Folate_DFE_mcg', 'Folate_food_mcg', 'Folate_total_mcg',
    'Folic_acid_mcg', 'Fructose_g', 'Galactose_g', 'Glucose_dextrose_g',
    'Glutamic_acid_g', 'Glycine_g', 'Histidine_g', 'Hydroxyproline_g',
    'Iron_Fe_mg', 'Isoleucine_g', 'Lactose_g', 'Leucine_g',
    'Lutein_+_zeaxanthin_mcg', 'Lycopene_mcg', 'Lysine_g', 'Magnesium_Mg_mg',
    'Maltose_g', 'Manganese_Mn_mg', 'Menaquinone_4_mcg', 'Methionine_g',
    'Niacin_mg', 'Pantothenic_acid_mg', 'Phenylalanine_g', 'Phosphorus_P_mg',
    'Phytosterols_mg', 'Potassium_K_mg', 'Proline_g', 'Protein_g',
    'Retinol_mcg', 'Riboflavin_mg', 'Selenium_Se_mcg', 'Serine_g',
    'Sodium_Na_mg', 'Starch_g', 'Stigmasterol_mg', 'Sucrose_g',
    'Sugars_total_g', 'Theobromine_mg', 'Thiamin_mg', 'Threonine_g',
    'Tocopherol_beta_mg', 'Tocopherol_delta_mg', 'Tocopherol_gamma_mg',
    'Tocotrienol_alpha_mg', 'Tocotrienol_beta_mg', 'Tocotrienol_delta_mg',
    'Tocotrienol_gamma_mg', 'Total_lipid_fat_g', 'Tryptophan_g', 'Tyrosine_g',
    'Valine_g', 'Vitamin_A_IU_IU', 'Vitamin_A_RAE_mcg',
    'Vitamin_B_12_added_mcg', 'Vitamin_B_12_mcg', 'Vitamin_B_6_mg',
    'Vitamin_C_total_ascorbic_acid_mg', 'Vitamin_D2_ergocalciferol_mcg',
    'Vitamin_D3_cholecalciferol_mcg', 'Vitamin_D_D2_+_D3_mcg', 'Vitamin_D_IU',
    'Vitamin_E_added_mg', 'Vitamin_E_alpha_tocopherol_mg',
    'Vitamin_K_phylloquinone_mcg', 'Water_g', 'Zinc_Zn_mg'
]
NUTRIENT_LIST_WITH_INDEX = [
    (0, '10:0_carpic_acid_g'),
    (1, '12:0_Lauric_Acid_g'),
    (2, '13:0_Tridecanoic_acid_g'),
    (3, '14:0_Miristic_Acid_g'),
    (4, '14:1_Miristoleic_acid_g'),
    (5, '15:0_Pentadecanoic_acid_g'),
    (6, '15:1_g'),
    (7, '16:0_g'),
    (8, '16:1_c_g'),
    (9, '16:1_t_g'),
    (10, '16:1_undifferentiated_g'),
    (11, '17:0_g'),
    (12, '17:1_g'),
    (13, '18:0_g'),
    (14, '18:1_11_t_18:1t_n_7_g'),
    (15, '18:1_c_g'),
    (16, '18:1_t_g'),
    (17, '18:1_undifferentiated_g'),
    (18, '18:2_CLAs_g'),
    (19, '18:2_i_g'),
    (20, '18:2_n_6_c_Linoleic_acid_omega_6c_g'),
    (21, '18:2_t_not_further_defined_g'),
    (22, '18:2_t_t_g'),
    (23, '18:2_undifferentiated_g'),
    (24, '18:3_n_3_ccc_alpha_Linolenic_acid_omega_3_ALA_g'),
    (25, '18:3_n_6_ccc_Gamma_linolenic_acid_omega_6_g'),
    (26, '18:3_undifferentiated_g'),
    (27, '18:3i_g'),
    (28, '18:4_g'),
    (29, '20:0_g'),
    (30, '20:1_g'),
    (31, '20:2_n_6_cc_g'),
    (32, '20:3_n_3_g'),
    (33, '20:3_n_6_g'),
    (34, '20:3_undifferentiated_g'),
    (35, '20:4_n_6_g'),
    (36, '20:4_undifferentiated_g'),
    (37, '20:5_n_3_EPAeicosapentaenoic_acid_omega_3_g'),
    (38, '21:5_g'),
    (39, '22:0_g'),
    (40, '22:1_c_g'),
    (41, '22:1_t_g'),
    (42, '22:1_undifferentiated_g'),
    (43, '22:4_g'),
    (44, '22:5_n_3_DPA_g'),
    (45, '22:6_n_3_DHA_docosahexaenoic_acid_omega_3_g'),
    (46, '24:0_g'),
    (47, '24:1_c_g'),
    (48, '4:0_g'),
    (49, '6:0_g'),
    (50, '8:0_g'),
    (51, 'Alanine_g'),
    (52, 'Alcohol_ethyl_g'),
    (53, 'Arginine_g'),
    (54, 'Ash_g'),
    (55, 'Aspartic_acid_g'),
    (56, 'Beta_sitosterol_mg'),
    (57, 'Betaine_mg'),
    (58, 'Caffeine_mg'),
    (59, 'Calcium_Ca_mg'),
    (60, 'Campesterol_mg'),
    (61, 'Carbohydrate_by_difference_g'),
    (62, 'Carotene_alpha_mcg'),
    (63, 'Carotene_beta_mcg'),
    (64, 'Cholesterol_mg'),
    (65, 'Choline_total_mg'),
    (66, 'Copper_Cu_mg'),
    (67, 'Cryptoxanthin_beta_mcg'),
    (68, 'Cystine_g'),
    (69, 'Dihydrophylloquinone_mcg'),
    (70, 'Energy_kJ'),
    (71, 'Energy_kcal'),
    (72, 'monounsaturated_Fatty_acids_total_g'),
    (73, 'polyunsaturated_Fatty_acids_total_g'),
    (74, 'saturated_Fatty_acids_total_g'),
    (75, 'trans_Fatty_acids_total_g'),
    (76, 'trans_monoenoic_Fatty_acids_total_g'),
    (77, 'trans_polyenoic_Fatty_acids_total_g'),
    (78, 'Fiber_total_dietary_g'),
    (79, 'Fluoride_F_mcg'),
    (80, 'Folate_DFE_mcg'),
    (81, 'Folate_food_mcg'),
    (82, 'Folate_total_mcg'),
    (83, 'Folic_acid_mcg'),
    (84, 'Fructose_g'),
    (85, 'Galactose_g'),
    (86, 'Glucose_dextrose_g'),
    (87, 'Glutamic_acid_g'),
    (88, 'Glycine_g'),
    (89, 'Histidine_g'),
    (90, 'Hydroxyproline_g'),
    (91, 'Iron_Fe_mg'),
    (92, 'Isoleucine_g'),
    (93, 'Lactose_g'),
    (94, 'Leucine_g'),
    (95, 'Lutein_+_zeaxanthin_mcg'),
    (96, 'Lycopene_mcg'),
    (97, 'Lysine_g'),
    (98, 'Magnesium_Mg_mg'),
    (99, 'Maltose_g'),
    (100, 'Manganese_Mn_mg'),
    (101, 'Menaquinone_4_mcg'),
    (102, 'Methionine_g'),
    (103, 'Niacin_mg'),
    (104, 'Pantothenic_acid_mg'),
    (105, 'Phenylalanine_g'),
    (106, 'Phosphorus_P_mg'),
    (107, 'Phytosterols_mg'),
    (108, 'Potassium_K_mg'),
    (109, 'Proline_g'),
    (110, 'Protein_g'),
    (111, 'Retinol_mcg'),
    (112, 'Riboflavin_mg'),
    (113, 'Selenium_Se_mcg'),
    (114, 'Serine_g'),
    (115, 'Sodium_Na_mg'),
    (116, 'Starch_g'),
    (117, 'Stigmasterol_mg'),
    (118, 'Sucrose_g'),
    (119, 'Sugars_total_g'),
    (120, 'Theobromine_mg'),
    (121, 'Thiamin_mg'),
    (122, 'Threonine_g'),
    (123, 'Tocopherol_beta_mg'),
    (124, 'Tocopherol_delta_mg'),
    (125, 'Tocopherol_gamma_mg'),
    (126, 'Tocotrienol_alpha_mg'),
    (127, 'Tocotrienol_beta_mg'),
    (128, 'Tocotrienol_delta_mg'),
    (129, 'Tocotrienol_gamma_mg'),
    (130, 'Total_lipid_fat_g'),
    (131, 'Tryptophan_g'),
    (132, 'Tyrosine_g'),
    (133, 'Valine_g'),
    (134, 'Vitamin_A_IU_IU'),
    (135, 'Vitamin_A_RAE_mcg'),
    (136, 'Vitamin_B_12_added_mcg'),
    (137, 'Vitamin_B_12_mcg'),
    (138, 'Vitamin_B_6_mg'),
    (139, 'Vitamin_C_total_ascorbic_acid_mg'),
    (140, 'Vitamin_D2_ergocalciferol_mcg'),
    (141, 'Vitamin_D3_cholecalciferol_mcg'),
    (142, 'Vitamin_D_D2_+_D3_mcg'),
    (143, 'Vitamin_D_IU'),
    (144, 'Vitamin_E_added_mg'),
    (145, 'Vitamin_E_alpha_tocopherol_mg'),
    (146, 'Vitamin_K_phylloquinone_mcg'),
    (147, 'Water_g'),
    (148, 'Zinc_Zn_mg')
]
# REQUIRED_NUTRIENT_LIST = [
#     20, 24, 59, 61, 64, 71, 74, 75, 78, 89, 91, 92, 94, 97, 98, 102,
#     105, 115, 122, 131, 133, 134, 137, 138, 139, 143, 145, 146, 148
# ]

REQUIRED_NUTRIENT_LIST = [
    20, 24, 59, 61, 64, 71, 74, 75, 78, 89, 91, 92, 94, 97, 98,
    115, 122, 131, 134, 137, 138, 139, 143, 145, 146
]

DAILY_NUTRIENTS_LIMIT = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1000.0, 0.0, 300.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0,
    22.0, 22.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 2.3, 0.0, 15.1, 3.2, 0.0, 7.0, 0.0, 0.0, 6.5, 400.0, 0.0,
    0.0, 0.0, 3.2, 0.0, 0.0, 6.0, 700.0, 0.0, 3500.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 2300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.9, 0.0, 4.1, 10000.0, 0.0, 0.0, 2.6, 1.7, 90.0, 0.0, 0.0,
    0.0, 800.0, 0.0, 15.0, 120.0, 0.0, 15.0
]
SELECTED_FOOD_IDS = [
    '20130', '20131', '20011', '20316', '20647', '20090', '20061', '20064',
    '20648', '20080', '20004', '20008', '20012', '20314', '20029', '20028',
    '20105', '20031', '20118', '20038', '20124', '20035', '20040', '20058',
    '20062', '20066', '20067', '20140', '20075', '16157', '16115', '11003',
    '11052', '11080', '11090', '11109', '11124', '11165', '11205', '11620',
    '11209', '11215', '11216', '11233', '11253', '11260', '11270', '11278',
    '11282', '11304', '11670', '11362', '11422', '11429', '11457', '11696',
    '11601', '16022', '16027', '16056', '16062', '16144', '16076', '16112',
    '16078', '16080', '16130', '16167', '16087', '16085', '16101', '16108',
    '16426', '16135', '16133', '4602', '4589', '4593', '4529', '4047', '4583',
    '4053', '4055', '4042', '4514', '4037', '4058', '4536', '4531', '4584',
    '4513', '2044', '2046', '2047', '2065', '2002', '2004', '2005', '2006',
    '2009', '2010', '2011', '2014', '2018', '2019', '2022', '2024', '2025',
    '2027', '2028', '2029', '2030', '2033', '2037', '2042', '2043', '2050',
    '2069', '1145', '1045', '1229', '1124', '1123', '1125', '1230', '1175',
    '1211', '1117', '1118', '1116', '9500', '9021', '9037', '9040', '9042',
    '9050', '9060', '9061', '9070', '9077', '9078', '9087', '9421', '9089',
    '9107', '9111', '9132', '9139', '9144', '9145', '9148', '9150', '9159',
    '9164', '9176', '9184', '9190', '9200', '9226', '9236', '9252', '9266',
    '9542', '9279', '9286', '9297', '9298', '9302', '9307', '9311', '9313',
    '9316', '9321', '9322', '9218', '9326', '5062', '5053', '5020', '5080',
    '5011', '5105', '5137', '5023', '5332', '5025', '5027', '15265', '15002',
    '15001', '15003', '15004', '17005', '9033', '9163', '9079', '9094', '9110',
    '9165', '9246', '9260', '9292', '12061', '12087', '12100', '12006',
    '12115', '12104', '12108', '12117', '12119', '12220', '12120', '12131',
    '12142', '12147', '12151', '12014', '12021', '12023', '12036', '12154',
    '12155', '12174'
]
SELECTED_FOODS = {
    1: (
        'Cereal_Grains',
        [],
        {
            1: '20130^Barley flour or meal',
            2: '20131^Barley malt flour',
            3: '20011^Buckwheat flour, whole-groat',
            4: '20316^Corn flour, whole-grain, white',
            5: '20647^Millet flour',
            6: '20090^Rice flour, brown',
            7: '20061^Rice flour, white, unenriched',
            8: '20064^Rye flour, medium',
            9: '20648^Sorghum flour, whole-grain',
            10: '20080^Wheat flour, whole-grain ',
            11: '20004^Barley, hulled',
            12: '20008^Buckwheat',
            13: '20012^Bulgur, dry',
            14: '20314^Corn grain, white',
            15: '20029^Couscous, cooked',
            16: '20028^Couscous, dry',
            17: '20105^Macaroni, vegetable, enriched, dry',
            18: '20031^Millet, raw',
            19: '20118^Noodles, flat, crunchy',
            20: '20038^Oats ',
            21: '20124^Pasta, whole-wheat, dry',
            22: '20035^Quinoa, uncooked',
            23: '20040^Rice, brown, medium-grain, raw',
            24: '20058^Rice, white, steamed',
            25: '20062^Rye grain',
            26: '20066^Semolina, enriched',
            27: '20067^Sorghum grain',
            28: '20140^Spelt, uncooked',
            29: '20075^Wheat, soft white',
            30: '16157^Chickpea flour (besan)',
            31: '16115^Soy flour, full-fat, raw'
        }
    ),
    2: (
        'Vegetables',
        [],
        {
            1: '11003^Amaranth leaves, raw',
            2: '11052^Beans, snap, green, raw',
            3: '11080^Beets, raw',
            4: '11090^Broccoli, raw',
            5: '11109^Cabbage, raw',
            6: '11124^Carrots, raw',
            7: '11165^Coriander (cilantro) leaves, raw',
            8: '11205^Cucumber, with peel, raw',
            9: '11620^Drumstick pods, raw',
            10: '11209^Eggplant, raw',
            11: '11215^Garlic, raw',
            12: '11216^Ginger root, raw',
            13: '11233^Kale, raw',
            14: '11253^Lettuce, green leaf, raw',
            15: '11260^Mushrooms, white, raw',
            16: '11270^Mustard greens, raw',
            17: '11278^Okra, raw',
            18: '11282^Onions, raw',
            19: '11304^Peas, green, raw',
            20: '11670^Peppers, hot chili, green, raw',
            21: '11362^Potatoes, raw, skin',
            22: '11422^Pumpkin, raw',
            23: '11429^Radishes, raw',
            24: '11457^Spinach, raw',
            25: '11696^Tomatoes, yellow, raw',
            26: '11601^Yam, raw'
        }
    ),
    3: (
        'Legume_Products',
        [],
        {
            1: '16022^Beans, french, mature seeds, raw',
            2: '16027^Beans, kidney, all types, mature seeds, raw',
            3: '16056^Chickpeas (beans, bengal gram), mature seeds, raw',
            4: '16062^Cowpeas, common , mature seeds, raw',
            5: '16144^Lentils, pink or red, raw',
            6: '16076^Lupins, mature seeds, raw',
            7: '16112^Miso',
            8: '16078^Mothbeans, mature seeds, raw',
            9: '16080^Mung beans, mature seeds, raw',
            10: '16130^Okara',
            11: '16167^Peanut Butter, smooth ',
            12: '16087^Peanuts, all types, raw',
            13: '16085^Peas, green, split, mature seeds, raw',
            14: '16101^Pigeon peas (red gram), mature seeds, raw',
            15: '16108^Soybeans, mature seeds, raw',
            16: '16426^Tofu, raw, firm, prepared with calcium sulfate',
            17: '16135^Winged beans, mature seeds, raw',
            18: '16133^Yardlong beans, mature seeds, raw'
        }
    ),
    4: (
        'Fats_And_Oils',
        [],
        {
            1: '4602^Butter, light, stick, without salt',
            2: '4589^Fish cod liver',
            3: '4593^Fish salmon',
            4: '4529^almond Oil',
            5: '4047^coconut Oil',
            6: '4583^mustard Oil',
            7: '4053^olive Oil, salad or cooking',
            8: '4055^palm Oil',
            9: '4042^peanut Oil, salad or cooking',
            10: '4514^poppyseed Oil',
            11: '4037^rice bran Oil',
            12: '4058^sesame Oil, salad or cooking',
            13: '4536^sheanut Oil',
            14: '4531^soybean Oil lecithin',
            15: '4584^sunflower Oil, high oleic (70% and over)',
            16: '4513^Vegetable palm kernel'
        }
    ),
    5: (
        'Spices',
        [],
        {
            1: '2044^Basil, fresh',
            2: '2046^Mustard, prepared, yellow',
            3: '2047^Salt, table',
            4: '2065^Spearmint, fresh',
            5: '2002^anise seed',
            6: '2004^bay leaf',
            7: '2005^caraway seed',
            8: '2006^cardamom',
            9: '2009^chili powder',
            10: '2010^cinnamon, ground',
            11: '2011^cloves, ground',
            12: '2014^cumin seed',
            13: '2018^fennel seed',
            14: '2019^fenugreek seed',
            15: '2022^mace, ground',
            16: '2024^mustard seed, ground',
            17: '2025^nutmeg, ground',
            18: '2027^oregano, dried',
            19: '2028^paprika',
            20: '2029^parsley, dried',
            21: '2030^pepper, black',
            22: '2033^poppy seed',
            23: '2037^saffron',
            24: '2042^thyme, dried',
            25: '2043^turmeric, ground',
            26: '2050^Vanilla extract',
            27: '2069^Vinegar, balsam'
        }
    ),
    6: (
        'Dairy_And_Egg_Products',
        [],
        {
            1: '1145^Butter, without salt',
            2: '1045^Cheese food, cold pack, American',
            3: '1229^Cheese, white, queso blanco',
            4: '1124^Egg, white, raw, fresh',
            5: '1123^Egg, whole, raw, fresh',
            6: '1125^Egg, yolk, raw, fresh',
            7: '1230^Milk, buttermilk, fluid, whole',
            8: '1175^Milk, fluid, 1% fat, without added vitamin A and D',
            9: '1211^Milk, whole, 3.25% fat, no added vitamin A and D',
            10: '1117^Yogurt, plain, low fat',
            11: '1118^Yogurt, plain, skim milk',
            12: '1116^Yogurt, plain, whole milk'
        }
    ),
    7: (
        'Fruits',
        [],
        {
            1: '9500^Apples, raw, red delicious, with skin ',
            2: '9021^Apricots, raw',
            3: '9037^Avocados, raw, all commercial varieties',
            4: '9040^Bananas, raw',
            5: '9042^Blackberries, raw',
            6: '9050^Blueberries, raw',
            7: '9060^Carambola, (starfruit), raw',
            8: '9061^Carissa, (natal-plum), raw',
            9: '9070^Cherries, sweet, raw',
            10: '9077^Crabapples, raw',
            11: '9078^Cranberries, raw',
            12: '9087^Dates, deglet noor',
            13: '9421^Dates, medjool',
            14: '9089^Figs, raw',
            15: '9107^Gooseberries, raw',
            16: '9111^Grapefruit, raw, pink and red and white, all areas',
            17: '9132^Grapes, red or green , raw',
            18: '9139^Guavas, common, raw',
            19: '9144^Jackfruit, raw',
            20: '9145^Java-plum, (jambolan), raw',
            21: '9148^Kiwifruit, green, raw',
            22: '9150^Lemons, raw, without peel',
            23: '9159^Limes, raw',
            24: '9164^Litchis, raw',
            25: '9176^Mangos, raw',
            26: '9184^Melons, honeydew, raw',
            27: '9190^Mulberries, raw',
            28: '9200^Oranges, raw, all commercial varieties',
            29: '9226^Papayas, raw',
            30: '9236^Peaches, yellow, raw',
            31: '9252^Pears, raw',
            32: '9266^Pineapple, raw, all varieties',
            33: '9542^Plantains, green, raw',
            34: '9279^Plums, raw',
            35: '9286^Pomegranates, raw',
            36: '9297^Raisins, golden, seedless',
            37: '9298^Raisins, dark, seedless ',
            38: '9302^Raspberries, raw',
            39: '9307^Rhubarb, raw',
            40: '9311^Roselle, raw',
            41: '9313^Sapodilla, raw chikoo',
            42: '9316^Strawberries, raw',
            43: '9321^Sugar-apples, (sweetsop), raw custard apple',
            44: '9322^Tamarinds, raw',
            45: '9218^Tangerines, (mandarin oranges), raw',
            46: '9326^Watermelon, raw'
        }
    ),
    8: (
        'Meat_Products',
        [],
        {
            1: '5062^Chicken, skinless, boneless, meat only, raw',
            2: '5053^Chicken, back, meat only, raw',
            3: '5020^Chicken, giblets, raw',
            4: '5080^Chicken, leg, meat only, raw',
            5: '5011^Chicken, meat only, raw',
            6: '5105^Chicken, wing, meat only, raw',
            7: '5137^Chicken, capons, giblets, raw',
            8: '5023^Chicken, gizzard, all classes, raw',
            9: '5332^Chicken, ground, raw',
            10: '5025^Chicken, heart, all classes, raw',
            11: '5027^Chicken, liver, all classes, raw',
            12: '15265^Fish, Salmon, canned, no skin and bones',
            13: '15002^Fish, anchovy, european, canned in oil, drained solids',
            14: '15001^Fish, anchovy, european, raw',
            15: '15003^Fish, bass, fresh water, mixed species, raw',
            16: '15004^Fish, bass, striped, raw',
            17: '17005^Lamb, trimmed retail cuts, raw'
        }
    ),
    9: (
        'Dryfruits_and_Nuts',
        [],
        {
            1: '9033^Apricots, dried, sulfured, stewed, without added sugar',
            2: '9163^Blueberries, dried, sweetened',
            3: '9079^Cranberries, dried, sweetened',
            4: '9094^Figs, dried, uncooked',
            5: '9110^Goji berries, dried',
            6: '9165^Litchis, dried',
            7: '9246^Peaches, dried, sulfured, uncooked',
            8: '9260^Pears, dried, sulfured, stewed, without added sugar',
            9: '9292^Plums, prunes, dried, stewed, without added sugar',
            10: '12061^almonds',
            11: '12087^cashew raw',
            12: '12100^chesteuropean, dried, peeled',
            13: '12006^chia dried',
            14: '12115^coconut cream, raw (liquid expressed from grated meat)',
            15: '12104^coconut meat, raw',
            16: '12108^coconut meat, dried (desiccated), not sweetened',
            17: '12117^coconut milk, raw ',
            18: '12119^coconut water (liquid from coconuts)',
            19: '12220^flax seed',
            20: '12120^hazelnuts or filberts',
            21: '12131^macadamia raw',
            22: '12142^pecans',
            23: '12147^pine dried',
            24: '12151^pistachio raw',
            25: '12014^pumpkin and squash seed kernels, dried',
            26: '12021^safflower seed kernels, dried',
            27: '12023^sesame whole, dried',
            28: '12036^sunflower seed kernels, dried',
            29: '12154^walblack, dried',
            30: '12155^walenglish',
            31: '12174^watermelon seed kernels, dried'
        }
    )
}

INPUT_FOOD_LIST = \
    "4047,4513,12108,1145,12115,12104,4602,12117,2043,1045,1229,17005,4536," \
    "2011,2042,1116,1230,1211,1117,2030,9144,1175,20008,2014,5062,20011," \
    "12087,9079,2009,12119,2005,2004,1125,20647,15265,20648,1123,1118,2046," \
    "12151,16085,2027,12014,2010,9542,11215,9087,11362,9040,9326,16144," \
    "5080,16130,5023,5027,5025,5137,5020,5011,20035,4593,4589,4584,4583," \
    "9033,5332,9021,9145,20028,9139,9132,9111,9110,9107,9094,9089,20029," \
    "9077,9070,9061,9060,9050,9042,9037,4531,20038,4055,4529,2002,2019,2018," \
    "20066,20067,20075,2006,20080,20090,20105,2024,20118,20124,20130,20131," \
    "20140,1124,20314,20316,2022,2025,4514,2065,20040,4058,9150,4053,20058," \
    "4037,2069,2050,2028,2047,2044,20061,20062,2037,2033,20064,2029,9148," \
    "9159,11304,12131,12120,16135,16157,16167,16426,12100,20004,12061,12036," \
    "12023,12021,12006,11696,11670,11620,11601,11457,11429,12142,12147," \
    "16056,16112,16108,16101,16087,16080,16078,16076,16062,16027,12155," \
    "16133,15004,15003,15002,15001,12220,12174,11422,11282,9163,11278,9302," \
    "9297,9292,9286,9279,9266,9260,9252,9246,9236,9226,9218,9200,9190,9184," \
    "16115,9164,20012,9311,9313,11124,11270,11260,11253,11233,11216,11209," \
    "11205,11165,11109,9316,11090,11080,11052,11003,9500,9421,9322,9321," \
    "9307,20031,5053,5105,9078,4042,12154,9165,16022,9298,9176".split(',')
