import unittest
from alignment_core import chunk_items


def unit(text,start,end): return dict(text=text,start_time=start,end_time=end)


class AlignmentTests(unittest.TestCase):
    def test_currency_percentage_negation_and_text(self):
        text='Do not sell SGD$1,250.50 at 0.5%.'
        words=['Do','not','sell','SGD125050','at','05']
        units=[unit(word,index,index+.8) for index,word in enumerate(words)]
        items,audit=chunk_items(text,units,0,6,[dict(start=0,end=6,speaker='A')])
        self.assertEqual([i['alternatives'][0]['content'] for i in items],['Do','not','sell','SGD$1,250.50','at','0.5%','.'])
        self.assertTrue(all(i.get('speaker_label')=='A' for i in items if i['type']=='pronunciation'))

    def test_han_and_mixed(self):
        items,_=chunk_items('买fund不要卖', [unit(t,i,i+.8) for i,t in enumerate(['买','fund','不','要','卖'])],10,5,[dict(start=10,end=15,speaker='B')])
        self.assertEqual([i['alternatives'][0]['content'] for i in items],['买','fund','不','要','卖'])
        self.assertEqual(items[0]['start_time'],'10.0')

    def test_ambiguous_boundary_not_guessed(self):
        items,audit=chunk_items('yes',[unit('yes',0,1)],0,1,[dict(start=0,end=.6,speaker='A'),dict(start=.6,end=1,speaker='B')])
        self.assertNotIn('speaker_label',items[0])
        self.assertIn('start_time',items[0])

    def test_text_mismatch_keeps_words_without_invented_timestamps(self):
        items,audit=chunk_items('not sell',[unit('sell',0,1)],0,1,[])
        self.assertEqual([i['alternatives'][0]['content'] for i in items],['not','sell'])
        self.assertTrue(all('start_time' not in i for i in items))

    def test_invalid_unit_does_not_discard_other_alignment(self):
        items,_=chunk_items('yes yes',[unit('yes',0,.4),unit('yes',float('nan'),1)],0,1,[dict(start=0,end=1,speaker='A')])
        self.assertEqual(items[0]['speaker_label'],'A')
        self.assertNotIn('start_time',items[1])

    def test_shared_unit_not_split_into_fabricated_word_times(self):
        items,_=chunk_items('a b',[unit('ab',0,1)],0,1,[])
        self.assertTrue(all('start_time' not in item for item in items))

    def test_timestamp_start_after_chunk_is_rejected(self):
        items,_=chunk_items('yes',[unit('yes',1.01,1.02)],0,1,[])
        self.assertNotIn('start_time',items[0])


if __name__=='__main__': unittest.main()
