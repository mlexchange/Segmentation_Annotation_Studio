import { beforeEach, describe, expect, it } from 'vitest';
import { useReferenceGuideStore } from './referenceGuideStore';

const initial = useReferenceGuideStore.getState();

beforeEach(() => {
  useReferenceGuideStore.setState(initial, true);
});

const cls = (label: string, overrides: Partial<{ color: string; description: string; exampleCrops: string[] }> = {}) => ({
  label, color: '#000', description: '', exampleCrops: [], ...overrides,
});

describe('referenceGuideStore', () => {
  it('starts empty', () => {
    const s = useReferenceGuideStore.getState();
    expect(s.entries).toEqual([]);
    expect(s.notes).toBe('');
    expect(s.loadedFor).toBeNull();
  });

  it('setGuide replaces the whole guide', () => {
    useReferenceGuideStore.getState().setGuide([cls('Cell')], 'my notes', 'local:x.tif');
    const s = useReferenceGuideStore.getState();
    expect(s.entries).toEqual([cls('Cell')]);
    expect(s.notes).toBe('my notes');
    expect(s.loadedFor).toBe('local:x.tif');
  });

  it('addEntry appends a class', () => {
    useReferenceGuideStore.getState().addEntry(cls('Pore'));
    expect(useReferenceGuideStore.getState().entries).toEqual([cls('Pore')]);
  });

  it('updateEntry merges partial updates by index', () => {
    useReferenceGuideStore.getState().setGuide([cls('Pore'), cls('Wall')], '', 'x');
    useReferenceGuideStore.getState().updateEntry(1, { color: '#f00' });
    expect(useReferenceGuideStore.getState().entries[1]).toEqual(cls('Wall', { color: '#f00' }));
    expect(useReferenceGuideStore.getState().entries[0]).toEqual(cls('Pore'));
  });

  it('removeEntry removes by index', () => {
    useReferenceGuideStore.getState().setGuide([cls('Pore'), cls('Wall')], '', 'x');
    useReferenceGuideStore.getState().removeEntry(0);
    expect(useReferenceGuideStore.getState().entries).toEqual([cls('Wall')]);
  });

  it('setNotes updates notes only', () => {
    useReferenceGuideStore.getState().setGuide([cls('Pore')], 'old', 'x');
    useReferenceGuideStore.getState().setNotes('new');
    expect(useReferenceGuideStore.getState().notes).toBe('new');
    expect(useReferenceGuideStore.getState().entries).toEqual([cls('Pore')]);
  });

  it('clear resets everything', () => {
    useReferenceGuideStore.getState().setGuide([cls('Pore')], 'n', 'x');
    useReferenceGuideStore.getState().clear();
    const s = useReferenceGuideStore.getState();
    expect(s.entries).toEqual([]);
    expect(s.notes).toBe('');
    expect(s.loadedFor).toBeNull();
  });

  describe('applyGenerated', () => {
    it('appends newly-discovered classes', () => {
      useReferenceGuideStore.getState().applyGenerated([cls('Pore', { color: '#111' })]);
      expect(useReferenceGuideStore.getState().entries).toEqual([cls('Pore', { color: '#111' })]);
    });

    it('preserves an existing class description, refreshing color/crops', () => {
      useReferenceGuideStore.getState().setGuide(
        [cls('Pore', { description: 'lead wrote this', color: '#000' })], '', 'x',
      );
      useReferenceGuideStore.getState().applyGenerated([
        cls('Pore', { description: 'auto-generated', color: '#f00', exampleCrops: ['data:img1'] }),
      ]);
      expect(useReferenceGuideStore.getState().entries).toEqual([
        cls('Pore', { description: 'lead wrote this', color: '#f00', exampleCrops: ['data:img1'] }),
      ]);
    });

    it('preserves existing entries not part of the new generation', () => {
      useReferenceGuideStore.getState().setGuide([cls('Pore'), cls('Wall')], '', 'x');
      useReferenceGuideStore.getState().applyGenerated([cls('Pore')]);
      const labels = useReferenceGuideStore.getState().entries.map((e) => e.label);
      expect(labels).toContain('Wall');
    });

    it('matches existing classes by label case-insensitively (trimmed)', () => {
      useReferenceGuideStore.getState().setGuide(
        [cls('  pore  ', { description: 'kept' })], '', 'x',
      );
      useReferenceGuideStore.getState().applyGenerated([cls('Pore', { description: 'new' })]);
      expect(useReferenceGuideStore.getState().entries[0].description).toBe('kept');
    });

    it('uses a default color when neither generated nor existing has one', () => {
      useReferenceGuideStore.getState().applyGenerated([{ label: 'Pore', color: '', description: '', exampleCrops: [] }]);
      expect(useReferenceGuideStore.getState().entries[0].color).toBe('#1f77b4');
    });
  });
});
