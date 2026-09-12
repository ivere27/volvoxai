import assert from 'node:assert/strict';
import test from 'node:test';

import * as inferenceProfile from '../ts/index.js';
import * as fullProfile from '../ts/full.js';

test('browser exports follow their declared capability profiles', () => {
  assert.equal('VxTrainingServiceClient' in inferenceProfile, false);
  assert.equal('VxQuantizationServiceClient' in inferenceProfile, false);
  assert.equal('FullEngineHost' in inferenceProfile, false);

  assert.equal(typeof fullProfile.VxTrainingServiceClient, 'function');
  assert.equal(typeof fullProfile.VxQuantizationServiceClient, 'function');
  assert.equal(typeof fullProfile.FullEngineHost, 'function');
  assert.equal('EngineHost' in fullProfile, false);

  for (const profile of [inferenceProfile, fullProfile]) {
    assert.equal(typeof profile.VxPlanningServiceClient, 'function');
    assert.equal(typeof profile.RpcError, 'function');
    assert.equal('check' in profile, false);
    assert.equal(typeof profile.pb.CreateGraphPlanRequest, 'function');
    assert.equal(typeof profile.pb.ResolveGraphPlanRequest, 'function');
    assert.equal('DataType' in profile, false, 'schema enums live only under pb');
    assert.equal('NativeStatus' in profile, false, 'schema enums live only under pb');
    assert.equal('OperatorKind' in profile, false, 'engine vocabulary is not public');
    assert.equal(
      'OperatorKind' in profile.pb,
      false,
      'internal engine vocabulary is absent from generated public protobuf',
    );
  }
  assert.equal('backend' in inferenceProfile, false);
  assert.equal('backend' in fullProfile, false);
});
