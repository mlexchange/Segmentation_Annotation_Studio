import { describe, expect, it, beforeEach } from 'vitest';
import {
  isPredictionClassVisible,
  useLayerVisibilityStore,
} from '@/stores/layerVisibilityStore';

describe('layerVisibilityStore', () => {
  beforeEach(() => {
    useLayerVisibilityStore.setState({
      groups: {
        image: true,
        features: true,
        proba: true,
        predictions: true,
        annotations: true,
        manifold: true,
      },
      predictionClassVisible: {},
      showPredictionMulti: true,
      showPredictionAbstain: true,
    });
  });

  it('toggles groups and prediction classes', () => {
    const s = useLayerVisibilityStore.getState();
    s.toggleGroup('proba');
    expect(useLayerVisibilityStore.getState().groups.proba).toBe(false);
    s.ensurePredictionClasses([1, 2]);
    expect(isPredictionClassVisible(useLayerVisibilityStore.getState().predictionClassVisible, 1)).toBe(
      true,
    );
    s.togglePredictionClass(1);
    expect(isPredictionClassVisible(useLayerVisibilityStore.getState().predictionClassVisible, 1)).toBe(
      false,
    );
  });
});
