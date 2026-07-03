import unittest

from src.ingestion.cleaner import DocumentCleaner

class TestDocumentCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = DocumentCleaner()

    def test_empty_text(self):
        self.assertEqual(self.cleaner.clean_markdown(""), "")
        self.assertEqual(self.cleaner.clean_markdown(None), "")

    def test_remove_html_anchors(self):
        raw_text = "# What is Amazon VPC?\n<a name=\"what-is-amazon-vpc\"></a>\n\nSome text."
        expected = "# What is Amazon VPC?\n\nSome text."
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

    def test_remove_generic_html(self):
        raw_text = "<p>Hello <b>world</b>!</p>"
        expected = "Hello world!"
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

    def test_clean_images(self):
        raw_text = "Here is a diagram: ![A VPC diagram](http://example.com/image.png) and some text."
        expected = "Here is a diagram: A VPC diagram and some text."
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

    def test_clean_links(self):
        raw_text = "For more details, see [Amazon VPC Pricing](https://aws.amazon.com/vpc/pricing/)."
        expected = "For more details, see Amazon VPC Pricing."
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

    def test_clean_links_and_images_combined(self):
        raw_text = "Check ![logo](img.png) at [AWS](https://aws.com)."
        expected = "Check logo at AWS."
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

    def test_newline_normalization(self):
        raw_text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
        expected = "Line 1\n\nLine 2\n\nLine 3"
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

    def test_trailing_whitespace(self):
        raw_text = "Line 1   \nLine 2 \t\nLine 3"
        expected = "Line 1\nLine 2\nLine 3"
        self.assertEqual(self.cleaner.clean_markdown(raw_text), expected)

if __name__ == '__main__':
    unittest.main()
