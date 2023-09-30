from chai.embeddings import get_summaries
from fixtures.reviews import full_text

def test_summary():

    summaries = get_summaries(
        text = full_text
    )

    for summary in summaries:
        print(summary["text"])

    assert summaries
    assert len(summaries) > 0