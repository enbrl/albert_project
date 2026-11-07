"""Tests of the information saved with a model (scripts/models.py) and its use by scripts/predict.py."""

import json

import pytest
import sklearn

from scripts import predict
from scripts.cli import UserError
from scripts.models import (
    compatibility_problem,
    describe_model,
    library_versions,
    main,
    make_metadata,
    metadata_path,
    read_metadata,
    save_model,
    train_and_evaluate,
    train_final_model,
)
from tests.test_models import make_cars

ARGUMENTS = ['--name', 'Peugeot 208', '--fuel', 'Essence', '--gearbox', 'Manuelle', '--km', '60000', '--year', '2019']


@pytest.fixture(scope='module')
def cars():
    return make_cars(200, seed=5)


@pytest.fixture(scope='module')
def trained(cars):
    model, metrics = train_and_evaluate(cars, kind='ridge')
    return train_final_model(cars, kind='ridge'), metrics


def other_version():
    """A version of scikit-learn that is not the installed one."""
    major, minor = (int(part) for part in sklearn.__version__.split('.')[:2])
    return f'{major}.{minor + 3}.0'


def test_metadata_path():
    assert metadata_path('models/price_model.joblib') == 'models/price_model.json'
    assert metadata_path('price.joblib') == 'price.json'
    assert metadata_path('a/b.c/model.pkl') == 'a/b.c/model.json'


def test_library_versions():
    versions = library_versions()
    assert set(versions) == {'python', 'scikit-learn', 'pandas', 'numpy', 'joblib'}
    assert versions['scikit-learn'] == sklearn.__version__


def test_make_metadata(cars, trained):
    _, metrics = trained
    metadata = make_metadata(cars, 'ridge', metrics, ['Marque', 'Km'])
    assert metadata['kind'] == 'ridge' and metadata['cars'] == 200 and metadata['columns'] == ['Marque', 'Km']
    assert len(metadata['created']) == 10 and metadata['created'][4] == '-'  # 2026-11-07
    assert metadata['metrics']['mae'] == pytest.approx(metrics['mae'], abs=0.001)
    json.dumps(metadata)  # it can be written as JSON


def test_save_model_writes_the_information_next_to_the_model(cars, trained, tmp_path):
    model, metrics = trained
    destination = tmp_path / 'models' / 'price.joblib'
    save_model(model, str(destination), make_metadata(cars, 'ridge', metrics, ['Km']))
    assert destination.exists() and (tmp_path / 'models' / 'price.json').exists()
    information = read_metadata(str(destination))
    assert information['kind'] == 'ridge' and information['cars'] == 200
    assert information['versions']['scikit-learn'] == sklearn.__version__


def test_save_model_without_information_still_saves_the_versions(trained, tmp_path):
    destination = tmp_path / 'price.joblib'
    save_model(trained[0], str(destination))
    assert read_metadata(str(destination))['versions']['scikit-learn'] == sklearn.__version__


def test_read_metadata_of_a_model_without_information(tmp_path):
    assert read_metadata(str(tmp_path / 'nothing.joblib')) is None
    broken = tmp_path / 'broken.json'
    broken.write_text('not json')
    assert read_metadata(str(tmp_path / 'broken.joblib')) is None
    (tmp_path / 'list.json').write_text('[1, 2]')
    assert read_metadata(str(tmp_path / 'list.joblib')) is None


def test_compatibility_problem():
    assert compatibility_problem({'versions': {'scikit-learn': sklearn.__version__}}) is None
    patch = sklearn.__version__.rsplit('.', 1)[0] + '.99'
    assert compatibility_problem({'versions': {'scikit-learn': patch}}) is None  # 1.6.1 and 1.6.99 are the same
    problem = compatibility_problem({'versions': {'scikit-learn': other_version()}})
    assert other_version() in problem and sklearn.__version__ in problem and 'train it again' in problem
    for nothing in (None, {}, {'versions': {}}, {'versions': {'scikit-learn': ''}}):
        assert compatibility_problem(nothing) is None


def test_describe_model(cars, trained):
    _, metrics = trained
    metadata = {**make_metadata(cars, 'ridge', metrics, ['Marque', 'Km']), 'versions': library_versions()}
    text = '\n'.join(describe_model(metadata))
    assert 'Model: ridge, trained on 200 cars' in text
    assert 'Columns: Marque, Km' in text and 'Mean error' in text and f'scikit-learn {sklearn.__version__}' in text
    assert 'Warning' not in text
    metadata['versions']['scikit-learn'] = other_version()
    assert 'Warning: this model was saved with scikit-learn' in '\n'.join(describe_model(metadata))


def test_describe_a_model_with_little_information():
    assert describe_model({})[0] == 'Model: ?, trained on ? cars on ?'


# ---------------------------------------------------------------- command line


def test_main_saves_the_information_and_info_shows_it(cars, tmp_path, capsys):
    csv = tmp_path / 'cars.csv'
    cars.to_csv(csv, index=False)
    destination = tmp_path / 'm.joblib'
    main([str(csv), '--kind', 'ridge', '--model', str(destination)])
    capsys.readouterr()
    information = read_metadata(str(destination))
    assert information['kind'] == 'ridge' and information['cars'] == 200
    assert information['metrics']['mae'] > 0 and 'Km' in information['columns']

    main(['--info', str(destination)])
    output = capsys.readouterr().out
    assert 'Model: ridge, trained on 200 cars' in output and 'Mean error' in output


def test_info_of_a_model_without_information(tmp_path):
    with pytest.raises(UserError, match='No information about the model'):
        main(['--info', str(tmp_path / 'nothing.joblib')])


def test_main_needs_data_or_info(capsys):
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2
    assert 'the data file is required' in capsys.readouterr().err


# ---------------------------------------------------------------- predict


def saved_model(trained, tmp_path, versions=None):
    destination = tmp_path / 'price.joblib'
    save_model(trained[0], str(destination))
    if versions:  # pretend the model was saved with other versions
        path = tmp_path / 'price.json'
        information = json.loads(path.read_text())
        information['versions'] = versions
        path.write_text(json.dumps(information))
    return str(destination)


def test_predict_is_silent_when_the_versions_match(trained, tmp_path, capsys):
    predict.main(['--model', saved_model(trained, tmp_path), *ARGUMENTS])
    assert capsys.readouterr().err == ''


def test_predict_warns_when_the_model_was_saved_with_another_version(trained, tmp_path, capsys):
    path = saved_model(trained, tmp_path, versions={'scikit-learn': other_version()})
    price = predict.main(['--model', path, *ARGUMENTS])  # it still works: it is a warning, not an error
    assert price > 0
    captured = capsys.readouterr()
    assert captured.err.startswith('Warning: this model was saved with scikit-learn ' + other_version())
    assert captured.out.startswith('Estimated price: ')  # the answer is not mixed with the warning


def test_predict_unreadable_model_tells_the_versions(trained, tmp_path, capsys):
    path = saved_model(trained, tmp_path, versions={'scikit-learn': other_version()})
    with open(path, 'wb') as file:
        file.write(b'broken')
    with pytest.raises(SystemExit) as stop:
        predict.main(['--model', path, *ARGUMENTS])
    assert stop.value.code == 1
    error = capsys.readouterr().err
    assert other_version() in error and sklearn.__version__ in error and 'cannot be read' in error


def test_predict_unreadable_model_without_information(tmp_path, capsys):
    broken = tmp_path / 'broken.joblib'
    broken.write_bytes(b'broken')
    with pytest.raises(SystemExit):
        predict.main(['--model', str(broken), *ARGUMENTS])
    assert 'another version of scikit-learn' in capsys.readouterr().err
