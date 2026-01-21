import IBM_PD_AD_v2 as v2

print("IBM_PD_AD_v2 Configuration Verification")
print("=" * 50)

# Basic parameters
print("\nBasic Parameters:")
print(f"  Population: {v2.general_config['population']:,}")
print(f"  Entrants per year: {v2.general_config['open_population']['entrants_per_year']:,}")
print(f"  Fixed entry age: {v2.general_config['open_population']['fixed_entry_age']}")
print(f"  Incidence growth rate: {v2.general_config['incidence_growth']['annual_rate']}")
print(f"  Reporting age bands: {v2.REPORTING_AGE_BANDS}")

# Severe_to_death RRs
print("\nSevere_to_death Relative Risks (all should be 1.00):")
rf = v2.general_config['risk_factors']
for factor in ['smoking', 'periodontal_disease', 'cerebrovascular_disease', 'CVD_disease', 'diabetes']:
    f_rr = rf[factor]['relative_risks']['severe_to_death']['female']
    m_rr = rf[factor]['relative_risks']['severe_to_death']['male']
    print(f"  {factor}: Female={f_rr}, Male={m_rr}")

# Initial prevalence
print("\nInitial Dementia Prevalence by Age Band:")
for band, vals in v2.general_config['initial_dementia_prevalence_by_age_band'].items():
    print(f"  {band}: Female={vals['female']:.6f}, Male={vals['male']:.6f}")

# Calculate expected prevalent cases
print("\nExpected Prevalent Cases Calculation:")
pop = v2.general_config['population']
age_weights = v2.general_config['initial_age_band_weights']
sex_dist = v2.general_config['sex_distribution']
prev = v2.general_config['initial_dementia_prevalence_by_age_band']

total_cases = 0
for band in [(65, 79), (80, 100)]:
    band_pop = pop * age_weights[band]
    female_pop = band_pop * sex_dist['female']
    male_pop = band_pop * sex_dist['male']
    band_cases = female_pop * prev[band]['female'] + male_pop * prev[band]['male']
    total_cases += band_cases
    print(f"  {band}: {band_cases:,.0f} cases")

print(f"\nTotal Expected Prevalent Cases: {total_cases:,.0f}")
print(f"Target: 800,000")
print(f"Difference: {total_cases - 800000:,.0f}")
