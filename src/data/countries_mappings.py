COUNTRY_OVERRIDES = {
    # Bolivia
    'plurinational state of': 'Bolivia',
    
    # Venezuela
    'bolivarian republic of': 'Venezuela',
    
    # South Korea
    'republic of korea': 'South Korea',

    # United States
    'usa': 'United States',
    'us': 'United States',
    'en:us': 'United States',
    'en:united-states': 'United States',
    'en:united states': 'United States',
    'états-unis': 'United States',
    'etats-unis': 'United States',
    'united states of america': 'United States',

    # United Kingdom
    'uk': 'United Kingdom',
    'en:gb': 'United Kingdom',
    'en:uk': 'United Kingdom',
    'en:united-kingdom': 'United Kingdom',
    'en:united kingdom': 'United Kingdom',
    'royaume-uni': 'United Kingdom',
    'great britain': 'United Kingdom',
    'england': 'United Kingdom',

    # Germany
    'de': 'Germany',
    'en:de': 'Germany',
    'en:germany': 'Germany',
    'deutschland': 'Germany',
    'allemagne': 'Germany',
    'alemania': 'Germany',

    # France
    'fr': 'France',
    'en:fr': 'France',
    'en:france': 'France',
    'francia': 'France',
    'frankreich': 'France',

    # Spain
    'es': 'Spain',
    'en:es': 'Spain',
    'en:spain': 'Spain',
    'españa': 'Spain',
    'espagne': 'Spain',
    'spanien': 'Spain',

    # Italy
    'it': 'Italy',
    'en:it': 'Italy',
    'en:italy': 'Italy',
    'italia': 'Italy',
    'italie': 'Italy',
    'italien': 'Italy',

    # Netherlands
    'nl': 'Netherlands',
    'en:nl': 'Netherlands',
    'en:netherlands': 'Netherlands',
    'nederland': 'Netherlands',
    'holland': 'Netherlands',
    'pays-bas': 'Netherlands',

    # Belgium
    'be': 'Belgium',
    'en:be': 'Belgium',
    'en:belgium': 'Belgium',
    'belgique': 'Belgium',
    'belgië': 'Belgium',
    'belgien': 'Belgium',

    # Switzerland
    'ch': 'Switzerland',
    'en:ch': 'Switzerland',
    'en:switzerland': 'Switzerland',
    'suisse': 'Switzerland',
    'schweiz': 'Switzerland',
    'svizzera': 'Switzerland',

    # Austria
    'at': 'Austria',
    'en:at': 'Austria',
    'en:austria': 'Austria',
    'österreich': 'Austria',
    'autriche': 'Austria',

    # Portugal
    'pt': 'Portugal',
    'en:pt': 'Portugal',
    'en:portugal': 'Portugal',

    # Czechia
    'cz': 'Czechia',
    'en:cz': 'Czechia',
    'česko': 'Czechia',
    'czech republic': 'Czechia',

    # Poland
    'pl': 'Poland',
    'en:pl': 'Poland',
    'en:poland': 'Poland',
    'polska': 'Poland',
    'pologne': 'Poland',

    # Sweden
    'se': 'Sweden',
    'en:se': 'Sweden',
    'en:sweden': 'Sweden',
    'sverige': 'Sweden',
    'suède': 'Sweden',

    # Norway
    'no': 'Norway',
    'en:no': 'Norway',
    'en:norway': 'Norway',
    'norge': 'Norway',
    'norvège': 'Norway',

    # Denmark
    'dk': 'Denmark',
    'en:dk': 'Denmark',
    'en:denmark': 'Denmark',
    'danmark': 'Denmark',
    'danemark': 'Denmark',

    # Finland
    'fi': 'Finland',
    'en:fi': 'Finland',
    'en:finland': 'Finland',
    'suomi': 'Finland',
    'finlande': 'Finland',

    # Russia
    'ru': 'Russia',
    'en:ru': 'Russia',
    'en:russia': 'Russia',
    'russian federation': 'Russia',
    'russie': 'Russia',

    # Japan
    'jp': 'Japan',
    'en:jp': 'Japan',
    'en:japan': 'Japan',
    'japon': 'Japan',

    # China
    'cn': 'China',
    'en:cn': 'China',
    'en:china': 'China',
    'chine': 'China',

    # Brazil
    'br': 'Brazil',
    'en:br': 'Brazil',
    'en:brazil': 'Brazil',
    'brasil': 'Brazil',
    'brésil': 'Brazil',

    # Canada
    'ca': 'Canada',
    'en:ca': 'Canada',
    'en:canada': 'Canada',

    # Australia
    'au': 'Australia',
    'en:au': 'Australia',
    'en:australia': 'Australia',
    'australie': 'Australia',

    # Mexico
    'mx': 'Mexico',
    'en:mx': 'Mexico',
    'en:mexico': 'Mexico',
    'méxico': 'Mexico',
    'mexique': 'Mexico',

    # Ireland
    'ie': 'Ireland',
    'en:ie': 'Ireland',
    'en:ireland': 'Ireland',
    'irlande': 'Ireland',

    # Greece
    'gr': 'Greece',
    'en:gr': 'Greece',
    'en:greece': 'Greece',
    'grèce': 'Greece',
    'grecia': 'Greece',

    # Romania
    'ro': 'Romania',
    'en:ro': 'Romania',
    'en:romania': 'Romania',
    'roumanie': 'Romania',

    # Hungary
    'hu': 'Hungary',
    'en:hu': 'Hungary',
    'en:hungary': 'Hungary',
    'magyarország': 'Hungary',
    'hongrie': 'Hungary',
}

"""
Function to normalize a country name to its canonical form.
"""
def normalize_country(country_name: str) -> str:
    if not country_name:
        return country_name

    country_lower = country_name.lower().strip()

    if country_lower in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[country_lower]

    return country_name.strip()
