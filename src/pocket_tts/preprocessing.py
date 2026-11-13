import re
import inflect

p = inflect.engine()

def _num2word(token: str) -> str:
    if not token.isnumeric():
        return token
    return str(p.number_to_words(token))

def normalize_money(text: str) -> str:
    # Matches things like $100, $5.50, $1,200
    def replace_match(match):
        amount = match.group(1).replace(',', '')
        try:
            if '.' in amount:
                dollars, cents = amount.split('.')
                dollar_str = _num2word(dollars)
                cent_str = _num2word(cents)
                return f"{dollar_str} dollars and {cent_str} cents"
            else:
                return f"{p.number_to_words(int(amount))} dollars" # type:ignore
        except ValueError:
            return match.group(0)

    return re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', replace_match, text)

def normalize_symbols(text: str) -> str:
    # Replace common symbols with spoken equivalents
    replacements = {
        r'&': ' and ',
        r'%': ' percent ',
        r'@': ' at ',
        r'\+': ' plus '
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def normalize_measurements_and_temps(text: str) -> str:
    # Temperatures: 72°F or 72F -> seventy-two degrees Fahrenheit
    text = re.sub(r'(-?\d+)\s*(?:°|deg(?:ree)?s?)\s*([FC])\b',
                  lambda m: f"{p.number_to_words(int(m.group(1)))} degrees {'Fahrenheit' if m.group(2)=='F' else 'Celsius'}", #type:ignore
                  text, flags=re.IGNORECASE)
   
    # Measurements: 5kg, 10cm, 60mph
    unit_map = {
        'kg': 'kilograms',
        'g': 'grams',
        'cm': 'centimeters',
        'm': 'meters',
        'km': 'kilometers',
        'mph': 'miles per hour',
        'kbps': 'kilobits per second'
    }
   
    def replace_unit(match: re.Match[str]) -> str:
        num = _num2word(match.group(1))
        unit = unit_map.get(match.group(2).lower(), match.group(2))
        return f"{num} {unit}"

    text = re.sub(r'\b(\d+)\s*(kg|g|cm|m|km|mph|kbps)\b', replace_unit, text, flags=re.IGNORECASE)
    return text

def normalize_abbreviations_and_titles(text: str) -> str:
    # Common honorifics and abbreviations
    abbreviations = {
        r'\bMr\.(?=\s)': 'Mister',
        r'\bMrs\.(?=\s)': 'Missus',
        r'\bDr\.(?=\s)': 'Doctor',
        r'\bProf\.(?=\s)': 'Professor',
        r'\bvs\.(?=\s)': 'versus',
        r'\betc\.(?=\s)': 'et cetera'
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    return text

def normalize_digital_identifiers(text: str) -> str:
    def replace_url(match: re.Match[str]) -> str:
        url = match.group(0)
        clean_url = re.sub(r'^https?://(?:www\.)?', '', url)
        return clean_url.replace('.', ' dot ').replace('/', ' slash ')

    def replace_email(match: re.Match[str]) -> str:
        email = match.group(0)
        return email.replace('.', ' dot ').replace('@', ' at ')

    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    text = re.sub(url_pattern, replace_url, text)

    email_pattern = r'[a-zA-Z0-9_\.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, replace_email, text)
    return text

def normalize_numbers(text: str) -> str:
    # Standalone numbers (do last so they don't break money/measurements)
    def replace_num(match: re.Match[str]) -> str:
        try:
            return _num2word(match.group(0))
        except ValueError:
            return match.group(0)

    return re.sub(pattern=r"\b\d+\b", repl=replace_num, string=text)

def text_normalize_all(text):
    text = normalize_money(text)
    text = normalize_measurements_and_temps(text)
    text = normalize_abbreviations_and_titles(text)
    text = normalize_digital_identifiers(text)
    text = normalize_numbers(text)
    text = normalize_symbols(text)
   
    # Clean up extra whitespace left by substitutions
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == '__main__':
    raw_texts = [
        "Dr. Smith bought 5kg of apples for $100.50 at 72°F.",
        "He did this after visiting https://example.com/shop and emailing support@example.com & saving 50%!"
    ]

    for text in raw_texts:
        print(text_normalize_all(text))