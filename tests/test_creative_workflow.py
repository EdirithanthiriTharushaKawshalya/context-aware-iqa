"""Style and delivery suitability are independent, using synthetic labels only."""
import csv

from PIL import Image
import pytest
import torch

from src.annotations import AnnotationStore
from src.context_classifier import ContextClassifier
from src.pipeline import PipelineConfig
from src.training_data import implementation_fingerprint
from scripts.prepare_human_labels import FIELDS
from scripts.train_context_classifier import evaluate, fit_creative_head
from tests.test_context_classifier import entry, pipeline_for


def constant_head(keep):
    model = ContextClassifier()
    with torch.no_grad():
        model.model.weight.zero_()
        model.model.bias.fill_(4. if keep else -4.)
    return model


@pytest.mark.parametrize('keep', [True, False])
def test_creative_style_does_not_replace_delivery_decision(tmp_path, keep):
    model = constant_head(keep)
    model.creative_classifier = constant_head(True)
    model.metadata = {'training_protocol': 'human_event_holdout_v1',
                      'implementation_sha256': implementation_fingerprint()}
    path = tmp_path/'creative.pt'
    model.save(path)
    sample = entry(500., motion=.8)
    pipeline = pipeline_for(sample)
    image = Image.new('RGB', (100,100))
    assert pipeline.run(image, PipelineConfig(shot_type='candid', decision_mode='rules')).classification == 'Review'
    result = pipeline.run(image, PipelineConfig(shot_type='candid', decision_mode='learned', model_path=str(path)))
    assert result.creative_style == 'Creative' and result.creative_probability > .9
    assert result.classification == ('Keep' if keep else 'Review')
    assert not any('Possible motion blur' in reason for reason in result.review_reasons)


def test_style_head_learns_independently_and_survives_reload(tmp_path):
    # Both styles include Keep and Review photos. A model must not equate
    # creative style with delivery suitability. No claim about real photos.
    entries = [entry(sharpness, background_motion=background)
               for sharpness in (2., 600.) for background in (.1, .9)]
    quality = [0, 0, 1, 1]
    styles = [0, 1, 0, 1]
    model = ContextClassifier().fit(entries, quality, epochs=120)
    model.tune_threshold(entries, quality)
    before = [model.predict(e)['classification'] for e in entries]
    model.fit_creative(entries, styles, entries, styles, epochs=120)
    assert [model.predict(e)['classification'] for e in entries] == before
    assert [model.predict(e)['creative_style'] for e in entries] == ['Standard', 'Creative', 'Standard', 'Creative']
    path = tmp_path/'model.pt'
    model.save(path)
    loaded = ContextClassifier.load(path)
    assert [loaded.predict(e) for e in entries] == [model.predict(e) for e in entries]
    model.fit(entries, quality, epochs=1)
    assert model.creative_classifier is None  # Retraining must not retain a stale style head.


def test_training_does_not_invent_style_labels_and_reports_creative_false_reviews():
    model = constant_head(False)
    model.creative_classifier = constant_head(True)
    sample = entry(500.)
    row = dict(target=1, image_id='creative', event_id='heldout', shot_type='candid',
               expert_score_usability='Keep', creative_style='Creative')
    sample['rule_classification'] = 'Review'
    report = evaluate(model, [row], [sample], [.1])['creative_photos']
    assert report['human_creative_keep_count'] == 1
    assert report['creative_keeps_sent_to_review'] == 1
    assert report['creative_keep_recall'] == 0
    assert report['style_accuracy'] == 1
    fresh = constant_head(True)
    partitions = {s: [dict(creative_style='')] for s in ('train', 'validation')}
    extracted = {s: ([sample], [.1]) for s in partitions}
    assert not fit_creative_head(fresh, partitions, extracted)['trained']
    assert fresh.predict(sample)['creative_style'] == 'Unknown'
    partitions['train'][0]['creative_style'] = 'Creative'
    assert 'both' in fit_creative_head(fresh, partitions, extracted)['reason']


def test_old_annotation_sheet_adds_style_without_replacing_judgments(tmp_path):
    path = tmp_path/'old.csv'
    old_fields = [field for field in FIELDS if field != 'creative_style']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(dict(image_id='photo', file_path='image.jpg', event_id='event',
                             expert_score_usability='Review', review_reason='focus', labeler_id='first'))
    store = AnnotationStore(path)
    assert store.rows[0]['creative_style'] == ''
    store.save_label('photo', 'Keep', 'photographer', creative_style='Creative')
    saved = AnnotationStore(path)
    assert saved.rows[0]['expert_score_usability'] == 'Keep'
    assert saved.rows[0]['creative_style'] == 'Creative'
    with pytest.raises(ValueError, match='style'):
        saved.save_label('photo', 'Keep', 'photographer', creative_style='Motion detected')


def test_invalid_creative_head_is_rejected(tmp_path):
    model = constant_head(True)
    model.creative_classifier = constant_head(True)
    model.creative_classifier.scale[0] = 0
    path = tmp_path/'invalid.pt'
    model.save(path)
    with pytest.raises(ValueError, match='Invalid classifier'):
        ContextClassifier.load(path)
