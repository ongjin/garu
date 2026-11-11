import unittest

from sns_sentences import align, region, split_of


class RegionTest(unittest.TestCase):
    def test_after_terminal_punctuation(self):
        text = '가방에쏙!파우치'
        self.assertEqual(region(text, text.index('파')), 'P')

    def test_terminal_followed_by_laughter_and_emoji(self):
        text = '좋아요!ᄏᄏ😀다음'
        self.assertEqual(region(text, text.index('다')), 'P')

    def test_ornament_without_terminal(self):
        text = '좋아요ᄒᄒ다음'
        self.assertEqual(region(text, text.index('다')), 'O')

    def test_plain_position(self):
        text = '맛있어요다음에'
        self.assertEqual(region(text, text.index('다')), 'C')


class SplitTest(unittest.TestCase):
    def test_document_buckets(self):
        self.assertIn(split_of('ESRW2500000005.1'), {'train', 'dev', 'test'})

    def test_align_requires_exact_cover(self):
        paragraph = {'form': '가방에 쏙! 파우치에 쏙!'}
        sentences = [{'form': '가방에 쏙!'}, {'form': '파우치에 쏙!'}]
        self.assertEqual(align(paragraph['form'], sentences), ['가방에 쏙!', '파우치에 쏙!'])
        self.assertIsNone(align(paragraph['form'], sentences[:1]))


if __name__ == '__main__':
    unittest.main()
