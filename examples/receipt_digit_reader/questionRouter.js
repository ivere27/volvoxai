/*
 * Rule-based question router, ported from the producer's question_router.py.
 *
 * The reader network never sees the question. This module turns a free-form
 * Korean or English question into a field selector, then indexes the record the
 * network produced. It is dependency-free and ships with the model, because
 * the graph alone cannot answer a question.
 *
 * The model reads exactly two things: the phone number's digits and the street
 * number's digits. Anything else must return an empty answer. A router that
 * guesses is worse than one that declines, because a plausible wrong number is
 * indistinguishable from a right one downstream.
 *
 * Two ordering rules carry over from the producer and matter here too:
 *
 * 1. Item, price, and store-name questions are matched *before* address. The
 *    Korean particle `로` is a substring of ordinary words (`합계로`,
 *    `제품으로`, `세로`), so an address test that ran first would swallow them.
 * 2. An address question only yields the street number when it actually asks
 *    for a number, and digit indices are range-checked, so an out-of-range
 *    ordinal declines instead of wrapping around to the other end.
 */

export const SUPPORTED_FAMILIES = Object.freeze(['phone', 'address']);

const ORDINALS = Object.freeze({
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5,
  sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10,
});
const ORDINAL_PATTERN = Object.keys(ORDINALS).join('|');

export function cleanText(value) {
  return String(value ?? '')
    .replace(/\n/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .normalize('NFC');
}

export function digitsOnly(value) {
  return (String(value ?? '').match(/\d/g) ?? []).join('');
}

export function normalizeAddressText(value) {
  return cleanText(value).replace(/(\d+(?:\s+\d+)+)$/, (match) => match.replace(/\s+/g, ''));
}

export function streetNumberFromAddress(address) {
  const match = normalizeAddressText(address).match(/(\d+(?:\s+\d+)*)\s*$/);
  return match ? digitsOnly(match[1]) : '';
}

const isPhone = (question, lower) => /\b(?:phone|telephone|tel)\b/.test(lower)
  || ['전화', '폰번호', '연락처'].some((key) => question.includes(key));

const isItem = (question, lower) => new RegExp(
  '\\b(?:item|items|product|products|price|prices|qty|quantity|total|totals|'
  + 'subtotal|amount|cost|purchased|bought)\\b',
).test(lower)
  || ['품목', '상품', '제품', '가격', '단가', '수량', '구매', '구입', '총액', '금액', '합계', '개수']
    .some((key) => question.includes(key));

const isStoreName = (question, lower) => new RegExp(
  '\\b(?:store name|shop name|merchant|business name|name of the (?:store|shop))\\b',
).test(lower)
  || ['상호', '가게 이름', '매장 이름', '점포 이름', '가게명', '매장명']
    .some((key) => question.includes(key));

const isAddress = (question, lower) => /\b(?:address|street|road|location)\b/.test(lower)
  || /\b(?:st|rd)\./.test(lower)
  || ['주소', '도로명', '위치', '번지'].some((key) => question.includes(key))
  || /[가-힣]{1,10}(?:길|로|대로)\s*\[?\?/.test(question);

/** 'phone', 'address', 'item', 'store', or 'other'. Only the first two answer. */
export function routeFamilyFromQuestion(question) {
  const text = cleanText(question);
  const lower = text.toLowerCase();
  if (isPhone(text, lower)) return 'phone';
  if (isItem(text, lower)) return 'item';
  if (isStoreName(text, lower)) return 'store';
  if (isAddress(text, lower)) return 'address';
  return 'other';
}

/** 'front_N', 'back_N', or 'phone_digit' when no index can be recovered. */
export function phoneOpFromQuestion(question) {
  const text = cleanText(question).toLowerCase();
  let op = 'phone_digit';
  const ordinal = text.match(new RegExp(`\\b(${ORDINAL_PATTERN})\\b`));
  if (ordinal) op = `front_${ORDINALS[ordinal[1]]}`;
  const front = text.match(/(?:front of|from the front|digit)\D*(\d+)/);
  if (front) op = `front_${front[1]}`;
  const back = text.match(/(?:from the end|from the back|from the right|last)\D*(\d+)/);
  if (back) op = `back_${back[1]}`;
  const frontKo = text.match(/앞에서\s*(\d+)\s*번째/);
  if (frontKo) op = `front_${frontKo[1]}`;
  const frontKoLoose = text.match(/(?:앞|앞자리)\s*(\d+)\s*(?:번째|번)?/);
  if (frontKoLoose) op = `front_${frontKoLoose[1]}`;
  const backKo = text.match(/뒤에서\s*(\d+)\s*번째/);
  if (backKo) op = `back_${backKo[1]}`;
  const backKoLoose = text.match(/(?:뒤|뒷자리|끝자리)\s*(\d+)\s*(?:번째|번)?/);
  if (backKoLoose) op = `back_${backKoLoose[1]}`;
  if (['from the back', 'from the end', 'from last'].some((key) => text.includes(key))) {
    const tail = text.match(new RegExp(`\\b(${ORDINAL_PATTERN})\\b`));
    if (tail) op = `back_${ORDINALS[tail[1]]}`;
  }
  if (op === 'phone_digit') {
    // "last digit", "끝자리", "마지막 숫자" carry an implicit ordinal of one.
    if (/\blast\b/.test(text) || ['끝자리', '마지막'].some((key) => text.includes(key))) {
      op = 'back_1';
    } else if (/\bfirst\b/.test(text) || text.includes('첫자리') || text.includes('첫 번째')) {
      op = 'front_1';
    }
  }
  return op;
}

/** 'street_no' when the question asks for the street number, else 'unsupported'. */
export function addressOpFromQuestion(question) {
  const text = cleanText(question);
  const lower = text.toLowerCase();
  if (text.includes('[?]') || text.includes('?]')) return 'street_no';
  if (['fill the blank', 'fill in the blank'].some((key) => lower.includes(key))) return 'street_no';
  if (['빈 칸', '빈칸'].some((key) => text.includes(key))) return 'street_no';
  if (/\b(?:street|road|building|house|block)\s*(?:number|no\.?|num)\b/.test(lower)) return 'street_no';
  if (/\bnumber\b.*\b(?:address|street|road)\b/.test(lower)
      || /\b(?:address|street|road)\b.*\bnumber\b/.test(lower)) return 'street_no';
  if (text.includes('번지')) return 'street_no';
  if (/(?:도로명|주소|위치|길|로)\s*(?:뒤|뒤의|끝|마지막)?\s*(?:에)?\s*숫자/.test(text)) return 'street_no';
  if (/숫자/.test(text) && ['주소', '도로명', '위치'].some((key) => text.includes(key))) return 'street_no';
  return 'unsupported';
}

/** Index an already-read record. Returns '' when the question cannot be served. */
export function answerFromRecord(question, phone, street) {
  const family = routeFamilyFromQuestion(question);
  if (family === 'address') {
    return addressOpFromQuestion(question) === 'street_no' ? (street ?? '') : '';
  }
  if (family === 'phone') {
    const match = phoneOpFromQuestion(question).match(/^(front|back)_(\d+)$/);
    if (!match || !phone) return '';
    const index = Number(match[2]);
    if (!Number.isInteger(index) || index < 1 || index > phone.length) return '';
    return match[1] === 'front' ? phone[index - 1] : phone[phone.length - index];
  }
  return '';
}
