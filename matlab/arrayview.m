function varargout = arrayview(varargin)
% ARRAYVIEW  View numeric arrays in ArrayView.
%
%   arrayview(A)                       view one array
%   arrayview(A, B, ...)               several arrays side by side (up to six)
%   arrayview(..., Name, Value)        options, see below
%   h = arrayview(...)                 handle: h.update(B), h.close()
%   [ha, hb] = arrayview(A, B)         one handle per array; ha.update(A2)
%                                      replaces only that array
%
%   Options (the same as Python's arrayview.view):
%     'name'        display name, or a cell array with one name per array
%                   (default: the variable names)
%     'overlay'     a mask array, or a cell array of masks, drawn on every
%                   array it fits
%     'rgb'         true to show an array whose first or last dimension has
%                   size 3 or 4 as a colour image
%     'mode'        'ortho', 'mosaic', 'qmri', 'qmri-ortho', 'qmri-mosaic'
%     'dims'        dimensions shown as x and y, e.g. [1 2]
%     'index'       position along every dimension, e.g. [1 1 12]
%     'cmap'        colormap name, e.g. 'gray', 'viridis'
%     'vmin','vmax' display range
%     'log'         true for log scale
%     'complex'     'mag', 'phase', 'real' or 'imag'
%     'voxel_size'  size of one step along each dimension, e.g. [1 1 2]
%     'window'      'native', 'browser', 'vscode' or 'none'
%     'port'        server port (default 8123)
%
%   'dims' and 'index' count from 1, as MATLAB does.
%
%   Requires arrayview installed in MATLAB's Python environment:
%       pip install arrayview
%       pyenv('Version', '/path/to/python')
%
%   For zero-copy (recommended for large arrays), arrayview() auto-enables
%   in-process Python. Call arrayview() before any other py.* call in your
%   session, or set it manually:
%       pyenv('ExecutionMode', 'InProcess')

    % --- Version guard ---
    if verLessThan('matlab', '9.11')  % 9.11 = R2021b
        error('arrayview:unsupportedVersion', ...
            'arrayview() requires MATLAB R2021b (9.11) or later.');
    end

    % --- Auto-configure in-process Python for zero-copy ---
    % Must happen before Python is first loaded in this session.
    pe = pyenv();
    if strcmp(pe.Status, 'NotLoaded')
        pyenv('ExecutionMode', 'InProcess');
    elseif strcmp(pe.ExecutionMode, 'OutOfProcess')
        warning('arrayview:outOfProcess', '%s', ...
            ['ArrayView: Python is running out-of-process — the array will be ' ...
             'copied (doubles memory).' newline 'To avoid this, restart MATLAB and call ' ...
             'arrayview() before any other py.* call.']);
    end

    % --- Split arguments: leading arrays, then Name/Value options ---
    nArrays = 0;
    while nArrays < nargin && (isnumeric(varargin{nArrays + 1}) || islogical(varargin{nArrays + 1}))
        nArrays = nArrays + 1;
    end
    if nArrays == 0
        error('arrayview:noArray', 'arrayview() needs at least one numeric array.');
    end
    if nArrays > 6
        error('arrayview:tooManyArrays', 'arrayview() shows at most six arrays, got %d.', nArrays);
    end
    defnames = cell(1, nArrays);
    for i = 1:nArrays
        defnames{i} = inputname(i);
        if isempty(defnames{i})
            defnames{i} = sprintf('Array %d', i);
        end
    end

    p = inputParser;
    addParameter(p, 'name', defnames);
    addParameter(p, 'port', 8123);
    addParameter(p, 'overlay', []);
    addParameter(p, 'rgb', false);
    addParameter(p, 'mode', '');
    addParameter(p, 'dims', []);
    addParameter(p, 'index', []);
    addParameter(p, 'cmap', '');
    addParameter(p, 'vmin', []);
    addParameter(p, 'vmax', []);
    addParameter(p, 'log', false);
    addParameter(p, 'complex', '');
    addParameter(p, 'voxel_size', []);
    addParameter(p, 'window', '');
    parse(p, varargin{nArrays + 1:end});
    o = p.Results;

    % --- Python modules ---
    try
        np = py.importlib.import_module('numpy');
    catch ME
        error('arrayview:numpyNotFound', ...
            ['NumPy not found in MATLAB''s Python environment.' newline ...
             'Install it with:' newline ...
             '    pip install numpy' newline ...
             'Then restart MATLAB. (Error: %s)'], ME.message);
    end
    try
        av = py.importlib.import_module('arrayview');
    catch ME
        error('arrayview:notInstalled', ...
            ['ArrayView Python package not found.' newline ...
             'Install it in MATLAB''s Python environment:' newline ...
             '    pip install arrayview' newline ...
             'Then restart MATLAB. (Error: %s)'], ME.message);
    end

    % --- Wrap arrays (zero-copy in in-process mode via buffer protocol) ---
    arrays = cell(1, nArrays);
    for i = 1:nArrays
        arrays{i} = np.asarray(varargin{i});
    end

    % --- Options -> Python keyword arguments ---
    names = cellstr(o.name);
    if nArrays == 1
        kw = {'name', names{1}};
    elseif numel(names) == nArrays
        kw = {'name', py.list(names)};
    else
        error('arrayview:nameCount', ...
            '''name'' needs one name per array (%d), got %d.', nArrays, numel(names));
    end
    kw = [kw, {'port', int32(o.port)}];
    if ~isempty(o.overlay)
        if iscell(o.overlay)
            masks = cellfun(@(m) np.asarray(m), o.overlay, 'UniformOutput', false);
            kw = [kw, {'overlay', py.list(masks)}];
        else
            kw = [kw, {'overlay', np.asarray(o.overlay)}];
        end
    end
    if o.rgb,                   kw = [kw, {'rgb', true}]; end
    if ~isempty(o.mode),        kw = [kw, {'mode', char(o.mode)}]; end
    % MATLAB counts dimensions and positions from 1, Python from 0.
    if ~isempty(o.dims),        kw = [kw, {'dims', py.tuple(num2cell(int32(o.dims(:)') - 1))}]; end
    if ~isempty(o.index),       kw = [kw, {'index', py.tuple(num2cell(int32(o.index(:)') - 1))}]; end
    if ~isempty(o.cmap),        kw = [kw, {'cmap', char(o.cmap)}]; end
    if ~isempty(o.vmin),        kw = [kw, {'vmin', double(o.vmin)}]; end
    if ~isempty(o.vmax),        kw = [kw, {'vmax', double(o.vmax)}]; end
    if o.log,                   kw = [kw, {'log', true}]; end
    if ~isempty(o.complex),     kw = [kw, {'complex', char(o.complex)}]; end
    if ~isempty(o.voxel_size),  kw = [kw, {'voxel_size', py.tuple(num2cell(double(o.voxel_size(:)')))}]; end
    if ~isempty(o.window),      kw = [kw, {'window', char(o.window)}]; end

    % --- Call arrayview.view() ---
    h = av.view(arrays{:}, pyargs(kw{:}));

    % --- Hand back the handle(s) when asked for ---
    if nargout == 0
        return
    end
    if nArrays == 1
        varargout{1} = h;
    elseif nargout == 1
        varargout{1} = h;              % Python tuple of handles
    else
        handles = cell(h);
        varargout = handles(1:nargout);
    end
end
