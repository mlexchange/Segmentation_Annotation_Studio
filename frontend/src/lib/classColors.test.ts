import { describe, it, expect, beforeEach } from 'vitest';
import { DEFAULT_COLORS, COLORBLIND_COLORS, getClassPalette, pickNextColor } from './classColors';
import { useSettingsStore } from '@/stores/settingsStore';
import { useClassStore } from '@/stores/classStore';

const HEX = /^#[0-9a-fA-F]{6}$/;

describe('class color palettes', () => {
  beforeEach(() => {
    useSettingsStore.setState({ colorblindMode: false });
    useClassStore.setState({ classes: [] });
  });

  it('both palettes are non-empty and valid 6-digit hex', () => {
    expect(DEFAULT_COLORS.length).toBeGreaterThan(0);
    expect(COLORBLIND_COLORS.length).toBeGreaterThan(0);
    for (const c of [...DEFAULT_COLORS, ...COLORBLIND_COLORS]) {
      expect(c).toMatch(HEX);
    }
  });

  it('colorblind palette has no duplicate colors', () => {
    expect(new Set(COLORBLIND_COLORS).size).toBe(COLORBLIND_COLORS.length);
  });

  it('getClassPalette follows the colorblindMode preference', () => {
    expect(getClassPalette()).toBe(DEFAULT_COLORS);
    useSettingsStore.setState({ colorblindMode: true });
    expect(getClassPalette()).toBe(COLORBLIND_COLORS);
  });

  it('pickNextColor returns the first unused color', () => {
    expect(pickNextColor([], 0)).toBe(DEFAULT_COLORS[0]);
    expect(pickNextColor([DEFAULT_COLORS[0]], 1)).toBe(DEFAULT_COLORS[1]);
  });

  it('pickNextColor cycles by index when every color is used', () => {
    const used = new Set(DEFAULT_COLORS);
    expect(pickNextColor(used, DEFAULT_COLORS.length)).toBe(DEFAULT_COLORS[0]);
    expect(pickNextColor(used, DEFAULT_COLORS.length + 1)).toBe(DEFAULT_COLORS[1]);
  });
});

describe('classStore.remapColors', () => {
  beforeEach(() => {
    useClassStore.setState({ classes: [] });
  });

  it('reassigns every class color from the palette in list order', () => {
    const { addClass, remapColors } = useClassStore.getState();
    addClass('air', '#000000');
    addClass('pore', '#111111');
    addClass('sample', '#222222');

    remapColors(COLORBLIND_COLORS);

    const colors = useClassStore.getState().classes.map((c) => c.color);
    expect(colors).toEqual([COLORBLIND_COLORS[0], COLORBLIND_COLORS[1], COLORBLIND_COLORS[2]]);
  });

  it('cycles the palette when there are more classes than colors', () => {
    const { addClass, remapColors } = useClassStore.getState();
    const palette = ['#aaaaaa', '#bbbbbb'];
    addClass('a', '#000000');
    addClass('b', '#000000');
    addClass('c', '#000000');

    remapColors(palette);

    const colors = useClassStore.getState().classes.map((c) => c.color);
    expect(colors).toEqual(['#aaaaaa', '#bbbbbb', '#aaaaaa']);
  });

  it('leaves colors unchanged when given an empty palette', () => {
    const { addClass, remapColors } = useClassStore.getState();
    addClass('a', '#123456');
    remapColors([]);
    expect(useClassStore.getState().classes[0].color).toBe('#123456');
  });
});
