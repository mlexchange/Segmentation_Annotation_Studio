import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import IngestDropzone from './IngestDropzone';

function jsonResponse(body: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => body, text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) } as Response;
}

function makeFile(name: string, content = 'x') {
  return new File([content], name, { type: 'image/tiff' });
}

/** The dropzone renders two hidden <input type="file"> — plain picker, then folder picker. */
function getInputs(container: HTMLElement) {
  const inputs = container.querySelectorAll('input[type="file"]');
  return { fileInput: inputs[0] as HTMLInputElement, dirInput: inputs[1] as HTMLInputElement };
}

function renderDropzone(overrides: Partial<React.ComponentProps<typeof IngestDropzone>> = {}) {
  const props = {
    serverUri: 'http://tiled.example',
    onBrowse: vi.fn(),
    onAnnotate: vi.fn(),
    ...overrides,
  };
  const utils = render(<IngestDropzone {...props} />);
  return { ...utils, props };
}

/** Route fetch calls by endpoint fragment; each handler may be sync or async. */
function makeFetchMock(overrides: {
  preflight?: (body: any) => Response | Promise<Response>;
  upload?: (fd: FormData) => Response | Promise<Response>;
  status?: (jobId: string, call: number) => Response | Promise<Response>;
} = {}) {
  let statusCalls = 0;
  return vi.fn(async (input: RequestInfo | URL, opts?: RequestInit) => {
    const url = String(input);
    if (url.includes('/api/ingest/preflight')) {
      const body = JSON.parse(opts!.body as string);
      if (overrides.preflight) return overrides.preflight(body);
      return jsonResponse({ container_exists: false, existing_count: 0, conflicts: [], suggested_container_path: `${body.container_path}_2` });
    }
    if (url.includes('/api/ingest/upload')) {
      const fd = opts!.body as FormData;
      if (overrides.upload) return overrides.upload(fd);
      return jsonResponse({ job_id: 'job-1' });
    }
    if (url.includes('/api/ingest/status/')) {
      const jobId = url.split('/api/ingest/status/')[1];
      statusCalls++;
      if (overrides.status) return overrides.status(jobId, statusCalls);
      return jsonResponse({ state: 'done', total: 1, done: 1, failed: 0, skipped: 0, errors: [], container_path: 'browse/foo' });
    }
    throw new Error(`Unhandled fetch: ${url}`);
  });
}

beforeEach(() => {
  vi.stubGlobal('fetch', makeFetchMock());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('IngestDropzone', () => {
  it('shows an error when picking only unsupported files', async () => {
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    // userEvent.upload() enforces the input's `accept` attribute (as a real browser
    // would), so an unsupported extension never reaches the file list. Use
    // fireEvent.change with a directly-assigned FileList to exercise the
    // component's own isSupported() rejection instead.
    const file = makeFile('notes.txt');
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fireEvent.change(fileInput);
    expect(await screen.findByText('No supported image files (TIFF, PNG, JPG, NPY).')).toBeInTheDocument();
  });

  it('picks a supported file, uploads with no conflicts, polls to completion, and shows Browse/Annotate actions', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        status: (_id, call) =>
          call === 1
            ? jsonResponse({ state: 'running', total: 1, done: 0, failed: 0, skipped: 0, errors: [], container_path: 'browse/scan1' })
            : jsonResponse({ state: 'done', total: 1, done: 1, failed: 0, skipped: 0, errors: [], container_path: 'browse/scan1' }),
      }),
    );
    const { container, props } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('scan1.tif'));

    await waitFor(() => expect(screen.getByText(/processed…/)).toBeInTheDocument());
    await waitFor(
      () => expect(screen.getByText(/Ingested 1 of 1/)).toBeInTheDocument(),
      { timeout: 3000 },
    );

    const browseBtn = screen.getByRole('button', { name: 'Browse this dataset' });
    await user.click(browseBtn);
    expect(props.onBrowse).toHaveBeenCalledWith('browse/scan1', 1);

    const annotateBtn = screen.getByRole('button', { name: 'Open in Annotate' });
    await user.click(annotateBtn);
    expect(props.onAnnotate).toHaveBeenCalledWith('browse/scan1', 'scan1');
  }, 10000);

  it('derives the destination container from the single file name when left at the default', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: (body) => {
          expect(body.container_path).toBe('browse/myscan');
          return jsonResponse({ container_exists: false, existing_count: 0, conflicts: [], suggested_container_path: 'browse/myscan_2' });
        },
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('myscan.tif'));
    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
  });

  it('shows the destination field only after expanding "Save uploaded images to"', async () => {
    renderDropzone();
    expect(screen.queryByPlaceholderText('browse/my_dataset')).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByText(/Save uploaded images to/));
    expect(screen.getByPlaceholderText('browse/my_dataset')).toBeInTheDocument();
  });

  it('shows the ConflictDialog when preflight reports conflicts, and Replace resumes the upload with on_conflict=replace', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: () =>
          jsonResponse({
            container_exists: true,
            existing_count: 3,
            conflicts: [{ filename: 'dup.tif', key: 'dup' }],
            suggested_container_path: 'browse/dup_2',
          }),
        upload: (fd) => {
          expect(fd.get('on_conflict')).toBe('replace');
          return jsonResponse({ job_id: 'job-2' });
        },
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('dup.tif'));

    expect(await screen.findByText('This dataset already has these images')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Replace the existing image/ }));

    await waitFor(() => expect(screen.queryByText('This dataset already has these images')).not.toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
  });

  it('Cancel on the ConflictDialog dismisses it without uploading', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: () =>
          jsonResponse({
            container_exists: true,
            existing_count: 1,
            conflicts: [{ filename: 'dup.tif', key: 'dup' }],
            suggested_container_path: 'browse/dup_2',
          }),
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('dup.tif'));

    expect(await screen.findByText('This dataset already has these images')).toBeInTheDocument();
    await user.click(screen.getByText('Cancel'));
    expect(screen.queryByText('This dataset already has these images')).not.toBeInTheDocument();
    expect(screen.queryByText(/processed…/)).not.toBeInTheDocument();
  });

  it('"Browse the existing dataset" on the conflict dialog calls onBrowse with the existing container and count', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: () =>
          jsonResponse({
            container_exists: true,
            existing_count: 5,
            conflicts: [{ filename: 'dup.tif', key: 'dup' }],
            suggested_container_path: 'browse/dup_2',
          }),
      }),
    );
    const { container, props } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('dup.tif'));

    expect(await screen.findByText('This dataset already has these images')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Browse the existing dataset/ }));
    expect(props.onBrowse).toHaveBeenCalledWith('browse/dup', 5);
    expect(screen.queryByText('This dataset already has these images')).not.toBeInTheDocument();
  });

  it('"Ingest to a new dataset" updates the destination and uploads to the suggested path', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: () =>
          jsonResponse({
            container_exists: true,
            existing_count: 1,
            conflicts: [{ filename: 'dup.tif', key: 'dup' }],
            suggested_container_path: 'browse/dup_2',
          }),
        upload: (fd) => {
          expect(fd.get('container_path')).toBe('browse/dup_2');
          expect(fd.get('on_conflict')).toBe('fail');
          return jsonResponse({ job_id: 'job-3' });
        },
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('dup.tif'));

    expect(await screen.findByText('This dataset already has these images')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Ingest to a new dataset/ }));
    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
    await user.click(screen.getByText(/Save uploaded images to/));
    expect(screen.getByDisplayValue('browse/dup_2')).toBeInTheDocument();
  });

  it('falls back to uploading anyway when the preflight request itself fails', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: () => jsonResponse('boom', false, 500),
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('scan.tif'));

    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('shows an error message when starting the upload itself fails', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        upload: () => jsonResponse('server exploded', false, 500),
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('scan.tif'));

    expect(await screen.findByText(/Could not start the upload: server exploded/)).toBeInTheDocument();
  });

  it('shows the fatal-failure state ("Nothing was ingested") when the job errors out', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        status: () =>
          jsonResponse({ state: 'error', total: 2, done: 0, failed: 2, skipped: 0, errors: [
            { filename: 'a.tif', kind: 'unreadable', message: '' },
            { filename: 'b.tif', kind: 'unreadable', message: '' },
          ], container_path: 'browse/bad' }),
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, [makeFile('a.tif'), makeFile('b.tif')]);

    await waitFor(
      () => expect(screen.getByText(/Nothing was ingested/)).toBeInTheDocument(),
      { timeout: 3000 },
    );
    expect(screen.getByText(/2 of 2 failed/)).toBeInTheDocument();
    expect(screen.getByText(/2 images could not be read as images/)).toBeInTheDocument();
  }, 10000);

  it('expands and collapses the filename list for a multi-file error group', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        status: () =>
          jsonResponse({ state: 'done', total: 2, done: 0, failed: 2, skipped: 0, errors: [
            { filename: 'a.tif', kind: 'unreadable', message: '' },
            { filename: 'b.tif', kind: 'unreadable', message: '' },
          ], container_path: 'browse/bad' }),
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, [makeFile('a.tif'), makeFile('b.tif')]);

    await waitFor(
      () => expect(screen.getByText(/2 images could not be read as images/)).toBeInTheDocument(),
      { timeout: 3000 },
    );
    expect(screen.queryByText('a.tif')).not.toBeInTheDocument();
    await user.click(screen.getByText('show filenames'));
    expect(screen.getByText('a.tif')).toBeInTheDocument();
    expect(screen.getByText('b.tif')).toBeInTheDocument();
    await user.click(screen.getByText('hide filenames'));
    expect(screen.queryByText('a.tif')).not.toBeInTheDocument();
  }, 10000);

  it('post-upload backstop: shows ConflictDialog when a completed job reports conflict errors', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        status: () =>
          jsonResponse({
            state: 'done',
            total: 1,
            done: 0,
            failed: 1,
            skipped: 0,
            errors: [{ filename: 'dup.tif', kind: 'conflict', message: '' }],
            container_path: 'browse/dup',
          }),
        preflight: () =>
          jsonResponse({ container_exists: false, existing_count: 0, conflicts: [], suggested_container_path: 'browse/dup_2' }),
      }),
    );
    const { container } = renderDropzone();
    const { fileInput } = getInputs(container);
    const user = userEvent.setup();
    await user.upload(fileInput, makeFile('dup.tif'));

    await waitFor(
      () => expect(screen.getByText('This dataset already has these images')).toBeInTheDocument(),
      { timeout: 3000 },
    );
  }, 10000);

  it('picking a folder derives the destination from the folder name via webkitRelativePath', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: (body) => {
          expect(body.container_path).toBe('browse/myfolder');
          return jsonResponse({ container_exists: false, existing_count: 0, conflicts: [], suggested_container_path: 'browse/myfolder_2' });
        },
      }),
    );
    const { container } = renderDropzone();
    const { dirInput } = getInputs(container);
    const file = makeFile('a.tif');
    Object.defineProperty(file, 'webkitRelativePath', { value: 'myfolder/a.tif' });
    const user = userEvent.setup();
    await user.upload(dirInput, file);
    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
  });

  it('dropping a plain FileList (no directory entries) uploads the supported files', async () => {
    const { container } = renderDropzone();
    const dropzone = screen.getByText(/Drag an image file or folder/).closest('[role="button"]') as HTMLElement;
    const file = makeFile('dropped.tif');
    const dataTransfer = { items: [], files: [file] };
    fireEvent.drop(dropzone, { dataTransfer });
    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
  });

  it('dropping a directory via webkitGetAsEntry recurses and uploads all nested files', async () => {
    vi.stubGlobal(
      'fetch',
      makeFetchMock({
        preflight: (body) => {
          expect(body.names.sort()).toEqual(['nested.tif', 'top.tif']);
          expect(body.container_path).toBe('browse/mydir');
          return jsonResponse({ container_exists: false, existing_count: 0, conflicts: [], suggested_container_path: 'browse/mydir_2' });
        },
      }),
    );
    const topFile = makeFile('top.tif');
    const nestedFile = makeFile('nested.tif');

    const nestedFileEntry = {
      isFile: true,
      isDirectory: false,
      file: (cb: (f: File) => void) => cb(nestedFile),
    };
    let subdirRead = false;
    const subdirEntry = {
      isFile: false,
      isDirectory: true,
      name: 'subdir',
      createReader: () => ({
        readEntries: (cb: (entries: any[]) => void) => {
          if (subdirRead) return cb([]);
          subdirRead = true;
          cb([nestedFileEntry]);
        },
      }),
    };
    const topFileEntry = {
      isFile: true,
      isDirectory: false,
      file: (cb: (f: File) => void) => cb(topFile),
    };
    let rootRead = false;
    const rootDirEntry = {
      isFile: false,
      isDirectory: true,
      name: 'mydir',
      createReader: () => ({
        readEntries: (cb: (entries: any[]) => void) => {
          if (rootRead) return cb([]);
          rootRead = true;
          cb([topFileEntry, subdirEntry]);
        },
      }),
    };

    const { container } = renderDropzone();
    const dropzone = screen.getByText(/Drag an image file or folder/).closest('[role="button"]') as HTMLElement;
    const dataTransfer = {
      items: [{ webkitGetAsEntry: () => rootDirEntry }],
      files: [],
    };
    fireEvent.drop(dropzone, { dataTransfer });
    await waitFor(() => expect(screen.getByText(/processed…|Ingested/)).toBeInTheDocument());
  });

  it('sets and clears the dragging visual state on dragenter/dragleave', () => {
    const { container } = renderDropzone();
    const dropzone = screen.getByText(/Drag an image file or folder/).closest('[role="button"]') as HTMLElement;
    expect(dropzone.className).toContain('border-white/20');
    fireEvent.dragEnter(dropzone);
    expect(dropzone.className).toContain('border-sky-400');
    fireEvent.dragLeave(dropzone);
    expect(dropzone.className).toContain('border-white/20');
  });
});
