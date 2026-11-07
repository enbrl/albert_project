"""The texts of the web application (app.py) in English and in French, without Streamlit: easy to test.

Every text has the same key in both languages, and the same ``{placeholders}`` (the tests check it). The data uses
French labels for fuels and gearboxes (``Essence``, ``Manuelle``): they are what the model knows, so they stay the
values of the choices; only what the user reads is translated (``label``, ``translate_table``).
"""

LANGUAGES = {'en': 'English', 'fr': 'Français'}
DEFAULT_LANGUAGE = 'en'

TEXTS = {
    'en': {
        'page_title': 'Used car price',
        'title': '🚗 Used car price',
        'caption': 'Estimate the price of a used car from a model trained on the aramisauto.com listing.',
        'language': 'Language',
        'files': 'Files',
        'model_file': 'Trained model',
        'data_file': 'Cleaned cars (optional)',
        'data_help': 'Gives the list of the models and the cars to compare with.',
        'make_files': 'Make them with `python -m scripts.pipeline`.',
        'no_model': 'There is no model in `{path}`. Train it first: `python -m scripts.models data/cars_clean.csv`',
        'unreadable': (
            'The model in `{path}` cannot be read ({kind}). It was probably saved with another '
            'version of scikit-learn: train it again with `python -m scripts.models data/cars_clean.csv`.'
        ),
        'name': 'Brand and model',
        'name_help': 'For example Renault Clio',
        'fuel': 'Fuel',
        'gearbox': 'Gearbox',
        'km': 'Mileage (km)',
        'year': 'Year of first registration',
        'power': 'Power (horsepower), 0 if you do not know',
        'power_help': 'It makes the estimate more precise when the model knows the power.',
        'battery': 'Battery capacity (kWh), electric cars only, 0 if you do not know',
        'battery_help': 'An electric car with a bigger battery is worth more: it makes the estimate more precise.',
        'trim': 'Trim (optional), for example Allure or GT Line',
        'trim_help': 'Only its first word counts: Allure Pack is an Allure.',
        'submit': 'Estimate the price',
        'empty_name': 'Write the brand and the model of the car.',
        'estimated_price': 'Estimated price',
        'mean_error': 'On cars it had not seen, this model is wrong by {error} on average.',
        'similar_title': 'Cars like this in the data: {name}',
        'similar_text': '{cars} cars, from {low} to {high} (median {median}). The closest to yours:',
    },
    'fr': {
        'page_title': "Prix d'une voiture d'occasion",
        'title': "🚗 Prix d'une voiture d'occasion",
        'caption': "Estimez le prix d'une voiture d'occasion avec un modèle entraîné sur les annonces d'aramisauto.com.",
        'language': 'Langue',
        'files': 'Fichiers',
        'model_file': 'Modèle entraîné',
        'data_file': 'Voitures nettoyées (facultatif)',
        'data_help': 'Donne la liste des modèles et les voitures à comparer.',
        'make_files': 'Créez-les avec `python -m scripts.pipeline`.',
        'no_model': "Il n'y a pas de modèle dans `{path}`. Entraînez-le d'abord : `python -m scripts.models data/cars_clean.csv`",
        'unreadable': (
            'Le modèle de `{path}` ne peut pas être lu ({kind}). Il a probablement été enregistré avec une autre '
            'version de scikit-learn : entraînez-le de nouveau avec `python -m scripts.models data/cars_clean.csv`.'
        ),
        'name': 'Marque et modèle',
        'name_help': 'Par exemple Renault Clio',
        'fuel': 'Carburant',
        'gearbox': 'Boîte de vitesse',
        'km': 'Kilométrage (km)',
        'year': 'Année de première mise en circulation',
        'power': 'Puissance (chevaux), 0 si vous ne la connaissez pas',
        'power_help': "Elle rend l'estimation plus précise quand le modèle connaît la puissance.",
        'battery': 'Capacité de la batterie (kWh), voitures électriques seulement, 0 si vous ne la connaissez pas',
        'battery_help': "Une voiture électrique avec une plus grande batterie vaut plus cher : cela rend l'estimation plus précise.",
        'trim': 'Finition (facultatif), par exemple Allure ou GT Line',
        'trim_help': 'Seul son premier mot compte : une Allure Pack est une Allure.',
        'submit': 'Estimer le prix',
        'empty_name': 'Écrivez la marque et le modèle de la voiture.',
        'estimated_price': 'Prix estimé',
        'mean_error': "Sur des voitures qu'il n'avait pas vues, ce modèle se trompe de {error} en moyenne.",
        'similar_title': 'Voitures comparables dans les données : {name}',
        'similar_text': '{cars} voitures, de {low} à {high} (médiane {median}). Les plus proches de la vôtre :',
    },
}

# What the user reads for each value of the data (the French labels of the data are the values)
LABELS = {
    'en': {
        'Essence': 'Petrol',
        'Diesel': 'Diesel',
        'Électrique': 'Electric',
        'Hybride': 'Hybrid',
        'Hybride rechargeable': 'Plug-in hybrid',
        'GPL': 'LPG',
        'Manuelle': 'Manual',
        'Automatique': 'Automatic',
    },
    'fr': {},  # the data is already in French: a value is shown as it is
}

# The headers of the table of the closest cars (the columns of scripts/estimator.py: SIMILAR_COLUMNS)
COLUMNS = {
    'en': {
        'Nom': 'Model',
        'année': 'Year',
        'Km': 'Mileage (km)',
        'Carburant': 'Fuel',
        'Boite de vitesse': 'Gearbox',
        'Prix': 'Price (€)',
    },
    'fr': {
        'Nom': 'Modèle',
        'année': 'Année',
        'Km': 'Kilométrage (km)',
        'Carburant': 'Carburant',
        'Boite de vitesse': 'Boîte de vitesse',
        'Prix': 'Prix (€)',
    },
}


def text(language, key, **values):
    """A text of the application in a language, with its {placeholders} filled (English if the language is unknown)."""
    texts = TEXTS.get(language, TEXTS[DEFAULT_LANGUAGE])
    return texts[key].format(**values) if values else texts[key]


def label(language, value):
    """What the user reads for a value of the data (a fuel, a gearbox): the value itself when it has no translation."""
    return LABELS.get(language, {}).get(value, value)


def translate_table(language, table):
    """The table of the closest cars with translated headers and values (a new table; the data is not changed)."""
    shown = table.copy()
    for column in ('Carburant', 'Boite de vitesse'):
        if column in shown.columns:
            shown[column] = shown[column].map(lambda value: label(language, value))
    return shown.rename(columns=COLUMNS.get(language, {}))
