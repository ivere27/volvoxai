import test from 'node:test';
import assert from 'node:assert/strict';

import {
  addressOpFromQuestion,
  answerFromRecord,
  phoneOpFromQuestion,
  routeFamilyFromQuestion,
  streetNumberFromAddress,
} from '../questionRouter.js';

const PHONE = '0123456';
const STREET = '999';

test('item and store questions are routed before address', () => {
  // `로` is a substring of ordinary words, so an address test that ran first
  // would swallow these and answer them with a street number.
  assert.equal(routeFamilyFromQuestion('합계로 얼마입니까?'), 'item');
  assert.equal(routeFamilyFromQuestion('제품으로 무엇을 샀습니까?'), 'item');
  assert.equal(routeFamilyFromQuestion('가게 이름이 무엇입니까?'), 'store');
  assert.equal(routeFamilyFromQuestion('What is the total amount?'), 'item');
});

test('unanswerable families return an empty answer', () => {
  for (const question of ['합계는 얼마입니까?', 'What is the store name?', 'Nice weather?']) {
    assert.equal(answerFromRecord(question, PHONE, STREET), '');
  }
});

test('an address question without a number request declines', () => {
  assert.equal(routeFamilyFromQuestion('가게 주소가 무엇입니까?'), 'address');
  assert.equal(addressOpFromQuestion('가게 주소가 무엇입니까?'), 'unsupported');
  assert.equal(answerFromRecord('가게 주소가 무엇입니까?', PHONE, STREET), '');
});

test('street number questions resolve to the street block', () => {
  for (const question of [
    'What is the street number in the store address?',
    '주소의 번지는 무엇입니까?',
    'Fill in the blank: Example Road [?]',
  ]) {
    assert.equal(answerFromRecord(question, PHONE, STREET), STREET, question);
  }
});

test('phone digit indices are one-based from either end', () => {
  assert.equal(phoneOpFromQuestion('가게 전화번호의 뒤에서 2번째 숫자는?'), 'back_2');
  assert.equal(answerFromRecord('가게 전화번호의 뒤에서 2번째 숫자는?', PHONE, STREET), '5');
  assert.equal(answerFromRecord('가게 전화번호의 앞에서 1번째 숫자는?', PHONE, STREET), '0');
  assert.equal(
    answerFromRecord("What is the first number of the store's phone number?", PHONE, STREET),
    '0',
  );
});

test('bare last/first phone questions carry an implicit ordinal of one', () => {
  assert.equal(phoneOpFromQuestion('전화번호 끝자리는?'), 'back_1');
  assert.equal(answerFromRecord('전화번호 끝자리는?', PHONE, STREET), '6');
  assert.equal(answerFromRecord("What is the last digit of the phone number?", PHONE, STREET), '6');
});

test('out-of-range phone indices decline instead of wrapping', () => {
  // Index 0 must not resolve to phone[-1], and an index past the end must not
  // wrap: a plausible wrong digit is indistinguishable from a right one.
  assert.equal(answerFromRecord('전화번호 앞에서 0번째 숫자는?', PHONE, STREET), '');
  assert.equal(answerFromRecord('전화번호 앞에서 99번째 숫자는?', PHONE, STREET), '');
  assert.equal(answerFromRecord('전화번호 끝자리는?', '', STREET), '');
});

test('street number extraction reads the trailing digit group', () => {
  assert.equal(streetNumberFromAddress('예시시 샘플구 가상로 999'), '999');
  assert.equal(streetNumberFromAddress('Example Road 45'), '45');
  assert.equal(streetNumberFromAddress('no digits here'), '');
});
