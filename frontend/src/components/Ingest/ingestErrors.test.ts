import { describe, it, expect } from 'vitest';
import { describeIngestError, summarizeIngestErrors, type IngestError } from './ingestErrors';

function err(filename: string, kind: IngestError['kind'], message = ''): IngestError {
  return { filename, kind, message };
}

describe('describeIngestError', () => {
  it('names the file and the reason', () => {
    expect(describeIngestError(err('a.tif', 'conflict'))).toBe(
      'a.tif — already exists in this dataset',
    );
  });

  it('shows the backend message for an unknown kind instead of generic copy', () => {
    const line = describeIngestError(err('a.tif', 'unknown', 'array is 4-D'));
    expect(line).toBe('a.tif — array is 4-D');
  });

  it('renders a legacy plain-string error verbatim', () => {
    // Older backends sent `errors: string[]`; must not collapse to a useless line.
    expect(describeIngestError('a.tif: 409: /browse/ds/a')).toBe('a.tif: 409: /browse/ds/a');
  });

  it('omits the dash for a job-level failure with no filename', () => {
    expect(describeIngestError(err('', 'unreachable'))).toBe('could not reach the Tiled server');
  });
});

describe('summarizeIngestErrors', () => {
  it('collapses a whole folder of identical failures into one row', () => {
    const errors = Array.from({ length: 690 }, (_, i) => err(`img_${i}.tif`, 'conflict'));
    const groups = summarizeIngestErrors(errors);

    expect(groups).toHaveLength(1);
    expect(groups[0].summary).toBe('690 images already exist in this dataset');
    expect(groups[0].filenames).toHaveLength(690);
  });

  it('keeps distinct reasons in separate rows, in first-seen order', () => {
    const groups = summarizeIngestErrors([
      err('a.tif', 'conflict'),
      err('b.tif', 'unreadable'),
      err('c.tif', 'conflict'),
    ]);

    expect(groups.map((g) => g.kind)).toEqual(['conflict', 'unreadable']);
    expect(groups[0].filenames).toEqual(['a.tif', 'c.tif']);
  });

  it('does not merge unknown failures that have different messages', () => {
    const groups = summarizeIngestErrors([
      err('a.tif', 'unknown', 'array is 4-D'),
      err('b.tif', 'unknown', 'array is 4-D'),
      err('c.tif', 'unknown', 'disk full'),
    ]);

    expect(groups).toHaveLength(2);
    expect(groups[0].summary).toBe('2 images could not be ingested — array is 4-D');
    expect(groups[1].summary).toBe('c.tif — disk full');
  });

  it('renders a single failure as itself, not as "1 image …"', () => {
    const [group] = summarizeIngestErrors([err('a.tif', 'conflict')]);
    expect(group.summary).toBe('a.tif — already exists in this dataset');
  });
});
