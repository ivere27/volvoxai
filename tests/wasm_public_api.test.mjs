import assert from 'node:assert/strict';
import test from 'node:test';

import * as inferenceProfile from '../ts/index.js';
import * as fullProfile from '../ts/full.js';

test('browser exports follow their declared capability profiles', () => {
  assert.equal('VxTrainingServiceClient' in inferenceProfile, false);
  assert.equal('VxQuantizationServiceClient' in inferenceProfile, false);
  // Authoring is a full-profile surface; inference consumes models only.
  assert.equal('VxPlanningServiceClient' in inferenceProfile, false);
  assert.equal('CreateGraphPlanRequest' in inferenceProfile.pb, false);
  assert.equal('ResolveGraphPlanRequest' in inferenceProfile.pb, false);
  assert.equal('FullEngineHost' in inferenceProfile, false);

  assert.equal(typeof fullProfile.VxTrainingServiceClient, 'function');
  assert.equal(typeof fullProfile.VxQuantizationServiceClient, 'function');
  assert.equal(typeof fullProfile.VxPlanningServiceClient, 'function');
  assert.equal(typeof fullProfile.pb.CreateGraphPlanRequest, 'function');
  assert.equal(typeof fullProfile.pb.ResolveGraphPlanRequest, 'function');
  assert.equal(typeof fullProfile.FullEngineHost, 'function');
  // volvoxai is the default package, so its host is also reachable under the
  // plain name; ts/full.ts exports FullEngineHost as EngineHost.
  assert.equal(fullProfile.EngineHost, fullProfile.FullEngineHost);

  for (const profile of [inferenceProfile, fullProfile]) {
    assert.equal(typeof profile.RpcError, 'function');
    assert.equal('check' in profile, false);
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
