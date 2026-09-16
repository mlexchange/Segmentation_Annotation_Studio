import { describe, expect, it, beforeEach } from 'vitest';
import {
  isPredictionClassVisible,
  isShapeOriginVisible,
  useLayerVisibilityStore,
} from '@/stores/layerVisibilityStore';

describe('layerVisibilityStore', () => {
  beforeEach(() => {
    useLayerVisibilityStore.setState({
      groups: {
        image: true,
        denoise: true,
        features: true,
        proba: true,
        predictions: true,
        annotations: true,
        manifold: true,
      },
      predictionClassVisible: {},
      showPredictionMulti: true,
      showPredictionAbstain: true,
      annotationOriginVisible: { human: true, predicted: true },
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

  it('denoise group defaults on and toggles independently', () => {
    const s = useLayerVisibilityStore.getState();
    expect(useLayerVisibilityStore.getState().groups.denoise).toBe(true);
    s.toggleGroup('denoise');
    expect(useLayerVisibilityStore.getState().groups.denoise).toBe(false);
    expect(useLayerVisibilityStore.getState().groups.image).toBe(true);
  });

  it('toggles predicted/human annotation visibility independently', () => {
    const s = useLayerVisibilityStore.getState();
    expect(isShapeOriginVisible(useLayerVisibilityStore.getState().annotationOriginVisible, 'predicted')).toBe(true);
    expect(isShapeOriginVisible(useLayerVisibilityStore.getState().annotationOriginVisible, undefined)).toBe(true);
    s.setAnnotationOriginVisible('predicted', false);
    expect(isShapeOriginVisible(useLayerVisibilityStore.getState().annotationOriginVisible, 'predicted')).toBe(false);
    // Human-drawn (origin undefined) is unaffected by hiding predicted shapes.
    expect(isShapeOriginVisible(useLayerVisibilityStore.getState().annotationOriginVisible, undefined)).toBe(true);
  });
});
