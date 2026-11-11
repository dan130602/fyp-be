import unittest
from formatting import format_fee
from transcript_adapter import transcript_document


class AdapterTests(unittest.TestCase):
    def test_numeric_symbols_reach_the_scorer(self):
        text='$1,200.50 -5% +3.2% S$200 \u00a530'
        document=transcript_document(text)
        words=[item['alternatives'][0]['content'] for item in document['results']['items'] if item['type']=='pronunciation']
        self.assertEqual(words,['$1,200.50','-5%','+3.2%','S$200','\u00a530'])
        self.assertEqual(document['results']['transcripts'][0]['transcript'],text)

    def test_formatting_preserves_compound_fractions(self):
        self.assertEqual(format_fee('a fee of half a percent'),'a fee of 0.5 percent')
        self.assertEqual(format_fee('one and a half a percent'),'one and a half a percent')
        self.assertEqual(format_fee('one-half a percent'),'one-half a percent')

    def test_no_changes_to_values_consent_or_negation(self):
        text='Do not buy. I do not consent. SGD 250,000 at -5%. \u4e0d\u540c\u610f\u4e70\u5165\u3002'
        self.assertEqual(format_fee(text),text)


if __name__=='__main__':
    unittest.main()
