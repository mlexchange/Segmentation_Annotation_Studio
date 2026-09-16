import { describe, expect, it } from 'vitest';
import { formatApiError } from './apiError';

describe('formatApiError', () => {
  it('formats the real 422 a bad patch size produces', () => {
    const body = JSON.stringify({
      detail: [{
        type: 'greater_than_equal',
        loc: ['body', 'model', 'dinov3_lora', 'hyperparams', 'image_size'],
        msg: 'Input should be greater than or equal to 224',
        input: 32,
        ctx: { ge: 224 },
      }],
    });

    expect(formatApiError(body)).toBe('image size: Input should be greater than or equal to 224');
  });

  it('passes through a plain string detail (HTTPException)', () => {
    expect(formatApiError(JSON.stringify({ detail: 'Another job is already running' })))
      .toBe('Another job is already running');
  });

  it('joins several validation errors', () => {
    const body = JSON.stringify({
      detail: [
        { loc: ['body', 'epochs'], msg: 'Input should be less than or equal to 500' },
        { loc: ['body', 'lr'], msg: 'Input should be greater than 0' },
      ],
    });

    expect(formatApiError(body)).toBe(
      'epochs: Input should be less than or equal to 500; lr: Input should be greater than 0',
    );
  });

  it('omits the field when loc carries no usable name', () => {
    expect(formatApiError(JSON.stringify({ detail: [{ msg: 'Something went wrong' }] })))
      .toBe('Something went wrong');
  });

  it('falls back to raw text for a non-JSON body', () => {
    expect(formatApiError('Internal Server Error')).toBe('Internal Server Error');
  });

  it('falls back for JSON that is not a FastAPI error', () => {
    expect(formatApiError(JSON.stringify({ oops: true }))).toBe('{"oops":true}');
  });

  it('uses the fallback for an empty body', () => {
    expect(formatApiError('', 'Request failed (500).')).toBe('Request failed (500).');
    expect(formatApiError('   ', 'Request failed (500).')).toBe('Request failed (500).');
  });

  it('truncates a runaway body', () => {
    expect(formatApiError('x'.repeat(5000)).length).toBe(500);
  });

  it('marks a truncated body with an ellipsis, not a silent cut', () => {
    const result = formatApiError('x'.repeat(5000));
    expect(result.endsWith('…')).toBe(true);
  });

  it('does not add an ellipsis when the body is short enough as-is', () => {
    expect(formatApiError('short message').endsWith('…')).toBe(false);
  });

  it('handles a detail that is a list of plain strings, not validation objects', () => {
    expect(formatApiError(JSON.stringify({ detail: ['Field A is required', 'Field B is invalid'] })))
      .toBe('Field A is required; Field B is invalid');
  });

  it('handles a single validation-error object, not wrapped in a list', () => {
    const body = JSON.stringify({
      detail: { loc: ['body', 'image_size'], msg: 'Input should be greater than or equal to 224' },
    });
    expect(formatApiError(body)).toBe('image size: Input should be greater than or equal to 224');
  });

  it('names a whole list element by its index when loc ends in a number', () => {
    const body = JSON.stringify({
      detail: [{ loc: ['body', 'sources', 2], msg: 'field required' }],
    });
    expect(formatApiError(body)).toBe('sources[2]: field required');
  });

  it('falls back to a bare "item N" when a numeric loc has no preceding field name', () => {
    const body = JSON.stringify({ detail: [{ loc: [3], msg: 'field required' }] });
    expect(formatApiError(body)).toBe('item 3: field required');
  });

  it('still finds the leaf field name when a numeric index sits before it, not after', () => {
    // The index (which source) isn't the leaf here — "color" is — so this must
    // keep reporting the same leaf-only name it always has, unaffected by the
    // numeric-leaf handling added for the case above.
    const body = JSON.stringify({
      detail: [{ loc: ['body', 'sources', 2, 'color'], msg: 'invalid color' }],
    });
    expect(formatApiError(body)).toBe('color: invalid color');
  });
});
