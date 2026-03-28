function arrayview(A, varargin)
% ARRAYVIEW  View a numeric array in the ArrayView browser viewer.
%
%   arrayview(A)                      view array, name taken from variable name
%   arrayview(A, 'name', 'MyName')    use custom display name
%   arrayview(A, 'port', 8200)        use custom port (default 8123)
%
%   Requires arrayview installed in MATLAB's Python environment:
%       % In your system Python or conda env:
%       pip install arrayview
%       % Then configure MATLAB to use it:
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

    % --- Parse arguments ---
    defname = inputname(1);
    if isempty(defname)
        defname = 'Array';
    end
    p = inputParser;
    addParameter(p, 'name', defname);
    addParameter(p, 'port', 8123);
    parse(p, varargin{:});

    % --- Input validation ---
    validateattributes(A, {'numeric', 'logical'}, {}, 'arrayview', 'A');

    % --- Wrap array (zero-copy in in-process mode via buffer protocol) ---
    try
        np = py.importlib.import_module('numpy');
    catch ME
        error('arrayview:numpyNotFound', ...
            ['NumPy not found in MATLAB''s Python environment.' newline ...
             'Install it with:' newline ...
             '    pip install numpy' newline ...
             'Then restart MATLAB. (Error: %s)'], ME.message);
    end
    arr = np.asarray(A);

    % --- Call arrayview.view() ---
    try
        av = py.importlib.import_module('arrayview');
    catch ME
        error('arrayview:notInstalled', ...
            ['ArrayView Python package not found.' newline ...
             'Install it in MATLAB''s Python environment:' newline ...
             '    pip install arrayview' newline ...
             'Then restart MATLAB. (Error: %s)'], ME.message);
    end

    av.view(arr, name=p.Results.name, port=int32(p.Results.port));
end
