// Custom runner.js that properly handles package loading before code execution
// Based on working Pyodide example provided by user

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

// Capture the original console.log to use for actual output
const originalConsoleLog = console.log;

// Buffer for collecting loading messages
let loadingMessages = [];

// Override console.log to capture loading messages during startup
console.log = function(...args) {
  const message = args.join(' ');
  // Detect various loading message patterns
  const isLoadingMessage = message.includes('Loading') || 
                          message.includes('Didn\'t find package') || 
                          message.includes('attempting to load from') || 
                          message.includes('micropip') || 
                          message.includes('packaging') ||
                          message.includes('Loaded nltk') ||
                          message.includes('regex') ||
                          message.includes('sqlite3') ||
                          message.includes('cdn.jsdelivr.net') ||
                          message.includes('.whl');
  
  if (isLoadingMessage) {
    loadingMessages.push(message);
    // Send loading messages to stderr instead of stdout
    console.error(`[LOADING_MSG] ${message}`);
  } else {
    // This is probably actual output, use original console.log
    originalConsoleLog(...args);
  }
};

const pyodide = await pyodideModule.loadPyodide();

// Temporarily override console.log during micropip loading
console.log = function(...args) {
  const message = args.join(' ');
  loadingMessages.push(message);
  console.error(`[MICROPIP_LOADING] ${message}`);
};

// Load micropip immediately and keep reference
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");

// Restore console.log after micropip loading is complete
console.log = originalConsoleLog;

// Log that startup is complete (to stderr to avoid JSON parsing issues)
console.error(`[STARTUP] Pyodide and micropip loaded. Captured ${loadingMessages.length} loading messages.`);

// Set up basic environment variables
try {
  const env_vars = (Deno.args[0] ?? "").split(",").filter(Boolean);
  for (const key of env_vars) {
    const val = Deno.env.get(key);
    if (val !== undefined) {
      pyodide.runPython(`
import os
os.environ[${JSON.stringify(key)}] = ${JSON.stringify(val)}
      `);
    }
  }
} catch (e) {
  console.error("Error setting environment variables in Pyodide:", e);
}

// Extract package names from import statements
function extractPackages(code) {
  const packages = new Set();
  const lines = code.split('\n');
  
  for (const line of lines) {
    const trimmed = line.trim();
    
    // Skip empty lines and comments
    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }
    
    // Only process actual import statements
    if (trimmed.startsWith('import ')) {
      // Handle "import package" or "import package.module"
      const afterImport = trimmed.slice(7).trim();
      const packageName = afterImport.split('.')[0].split(' as ')[0].split(';')[0].trim();
      if (packageName && !isBuiltinModule(packageName)) {
        // Special case: sklearn imports require scikit-learn package
        if (packageName === 'sklearn') {
          packages.add('scikit-learn');
        } else {
          packages.add(packageName);
        }
      }
    } else if (trimmed.startsWith('from ')) {
      // Handle "from package import ..."
      const afterFrom = trimmed.slice(5).trim();
      const packageName = afterFrom.split('.')[0].split(' ')[0].split(';')[0].trim();
      if (packageName && !isBuiltinModule(packageName)) {
        // Special case: sklearn imports require scikit-learn package
        if (packageName === 'sklearn') {
          packages.add('scikit-learn');
        } else {
          packages.add(packageName);
        }
      }
    }
  }
  
  return Array.from(packages);
}

// Check if a module is a built-in Python module
function isBuiltinModule(moduleName) {
  const builtins = ['sys', 'os', 'math', 're', 'json', 'time', 'collections', 
                   'itertools', 'functools', 'operator', 'copy', 'pickle',
                   'datetime', 'random', 'string', 'io', 'urllib', 'http'];
  return builtins.includes(moduleName);
}

// Load a single package using the working pattern from user's example
async function loadSinglePackage(packageName, code = "") {
  console.error(`[LOADING] Installing package: ${packageName}`);
  
  // Temporarily capture stdout during package loading
  const tempLoadingMessages = [];
  const originalLog = console.log;
  console.log = function(...args) {
    const message = args.join(' ');
    tempLoadingMessages.push(message);
  };
  
  try {
    // First try loadPackage for built-in Pyodide packages
    try {
      await pyodide.loadPackage(packageName);
      console.error(`[SUCCESS] Loaded ${packageName} via loadPackage`);
      
      // If this is NLTK, only download the resources actually needed by the code
      if (packageName === 'nltk' && code) {
        await loadNLTKResourcesForCode(code);
      }
      
      return true;
    } catch (loadError) {
      console.error(`[INFO] ${packageName} not in Pyodide distribution, trying micropip...`);
      
      // Use micropip.install() just like in the working example
      await micropip.install(packageName);
      console.error(`[SUCCESS] Installed ${packageName} via micropip`);
      
      // If this is NLTK, only download the resources actually needed by the code
      if (packageName === 'nltk' && code) {
        await loadNLTKResourcesForCode(code);
      }
      
      return true;
    }
  } catch (error) {
    console.error(`[FAILED] Could not install ${packageName}: ${error.message}`);
    return false;
  } finally {
    // Always restore console.log
    console.log = originalLog;
    if (tempLoadingMessages.length > 0) {
      console.error(`[DEBUG] Captured ${tempLoadingMessages.length} loading messages for ${packageName}`);
    }
  }
}

// Analyze code and load only the NLTK resources that are actually needed
async function loadNLTKResourcesForCode(code) {
  console.error(`[NLTK] Analyzing code for required resources...`);
  
  // Detect which NLTK resources are needed based on the code
  const resourcesNeeded = [];
  
  if (code.includes('word_tokenize') || code.includes('sent_tokenize')) {
    resourcesNeeded.push({
      name: 'punkt',
      url: 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip',
      path: '/home/pyodide/nltk_data/tokenizers/punkt.zip',
      extractTo: '/home/pyodide/nltk_data/tokenizers/',
      checkFile: '/home/pyodide/nltk_data/tokenizers/punkt/english.pickle',
      timeout: 20
    });
  }
  
  if (code.includes('stopwords')) {
    resourcesNeeded.push({
      name: 'stopwords',
      url: 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip',
      path: '/home/pyodide/nltk_data/corpora/stopwords.zip',
      extractTo: '/home/pyodide/nltk_data/corpora/',
      checkFile: '/home/pyodide/nltk_data/corpora/stopwords/english',
      timeout: 10
    });
  }
  
  if (code.includes('pos_tag')) {
    resourcesNeeded.push({
      name: 'averaged_perceptron_tagger',
      url: 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip',
      path: '/home/pyodide/nltk_data/taggers/averaged_perceptron_tagger.zip',
      extractTo: '/home/pyodide/nltk_data/taggers/',
      checkFile: '/home/pyodide/nltk_data/taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle',
      timeout: 15
    });
  }
  
  if (code.includes('WordNetLemmatizer') || code.includes('wordnet')) {
    resourcesNeeded.push({
      name: 'wordnet',
      url: 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip',
      path: '/home/pyodide/nltk_data/corpora/wordnet.zip',
      extractTo: '/home/pyodide/nltk_data/corpora/',
      checkFile: '/home/pyodide/nltk_data/corpora/wordnet/index.noun',
      timeout: 25
    });
  }
  
  if (resourcesNeeded.length === 0) {
    console.error(`[NLTK] No specific resources needed, skipping downloads`);
    return;
  }
  
  // Check which resources are already available
  const resourcesToDownload = [];
  for (const resource of resourcesNeeded) {
    const isAvailable = await checkNLTKResourceAvailable(resource.checkFile);
    if (isAvailable) {
      console.error(`[NLTK] Resource ${resource.name} already available, skipping`);
    } else {
      resourcesToDownload.push(resource);
    }
  }
  
  if (resourcesToDownload.length === 0) {
    console.error(`[NLTK] All required resources already available`);
    return;
  }
  
  console.error(`[NLTK] Found ${resourcesToDownload.length} resources to download: ${resourcesToDownload.map(r => r.name).join(', ')}`);
  
  try {
    // Setup NLTK data directory structure first
    const nltkSetupCode = `
import os
import nltk

# Create NLTK data directory structure
nltk_data_dir = '/home/pyodide/nltk_data'
tokenizers_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
corpora_dir = os.path.join(nltk_data_dir, 'corpora', 'stopwords')
taggers_dir = os.path.join(nltk_data_dir, 'taggers', 'averaged_perceptron_tagger')
corpora_wordnet_dir = os.path.join(nltk_data_dir, 'corpora', 'wordnet')

for dir_path in [tokenizers_dir, corpora_dir, taggers_dir, corpora_wordnet_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

print("NLTK directories created")
`;
    
    const setupResult = await pyodide.runPythonAsync(nltkSetupCode);
    console.error(`[NLTK] Setup: ${setupResult}`);
    
    // Download and extract only the needed resources that aren't already available
    for (const resource of resourcesToDownload) {
      await downloadNLTKResource(resource.url, resource.path);
      await extractNLTKResource(resource.path, resource.extractTo, resource.timeout);
    }
    
  } catch (error) {
    console.error(`[NLTK] Setup error (continuing anyway): ${error.message}`);
  }
}

// Check if an NLTK resource is already available
async function checkNLTKResourceAvailable(checkFile) {
  const checkCode = `
import os
result = os.path.exists("${checkFile}")
result
`;
  
  try {
    const result = await pyodide.runPythonAsync(checkCode);
    return result === true;
  } catch (error) {
    return false;
  }
}

// Extract a single NLTK resource with timeout protection
async function extractNLTKResource(zipPath, extractTo, timeout = 30) {
  const extractCode = `
import zipfile
import os
import time

def extract_with_timeout(zip_path, extract_to, timeout):
    try:
        start_time = time.time()
        if not os.path.exists(zip_path):
            return f"Zip file not found: {zip_path}"
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for i, member in enumerate(members):
                if time.time() - start_time > timeout:
                    return f"Timeout extracting {zip_path} after {timeout}s"
                try:
                    zip_ref.extract(member, extract_to)
                except Exception as e:
                    continue  # Skip problematic files
            return f"Extracted {zip_path} ({len(members)} files)"
    except Exception as e:
        return f"Error extracting {zip_path}: {str(e)}"

result = extract_with_timeout("${zipPath}", "${extractTo}", ${timeout})
result
`;
  
  try {
    const result = await pyodide.runPythonAsync(extractCode);
    console.error(`[NLTK] Extraction: ${result}`);
  } catch (error) {
    console.error(`[NLTK] Extraction error: ${error.message}`);
  }
}

// Download NLTK resource using fetch (following the Stack Overflow solution)
async function downloadNLTKResource(url, filePath) {
  try {
    console.error(`[NLTK] Downloading ${url} to ${filePath}`);
    
    const downloadCode = `
from js import fetch

async def download_resource(url, file_path):
    try:
        response = await fetch(url)
        if response.status != 200:
            return f"Failed to download {url}: HTTP {response.status}"
            
        js_buffer = await response.arrayBuffer()
        py_buffer = js_buffer.to_py()  # this is a memoryview
        stream = py_buffer.tobytes()  # now we have a bytes object
        
        # write to the appropriate path
        with open(file_path, "wb") as fh:
            fh.write(stream)
            
        return f"Successfully downloaded {url} ({len(stream)} bytes)"
    except Exception as e:
        return f"Error downloading {url}: {str(e)}"

result = await download_resource("${url}", "${filePath}")
result
`;
    
    const result = await pyodide.runPythonAsync(downloadCode);
    console.error(`[NLTK] ${result}`);
    
  } catch (error) {
    console.error(`[NLTK] Download error for ${url}: ${error.message}`);
  }
}

// Load all required packages BEFORE executing any code
async function loadRequiredPackages(code) {
  const packages = extractPackages(code);
  
  if (packages.length === 0) {
    return { success: true, loadedPackages: [], failedPackages: [] };
  }
  
  console.error(`[INFO] Found packages to load: ${packages.join(', ')}`);
  
  const loadedPackages = [];
  const failedPackages = [];
  
  // Load packages one by one and wait for each to complete
  for (const pkg of packages) {
    const success = await loadSinglePackage(pkg, code);
    if (success) {
      loadedPackages.push(pkg);
    } else {
      failedPackages.push(pkg);
    }
  }
  
  if (loadedPackages.length > 0) {
    console.error(`[SUCCESS] Loaded packages: ${loadedPackages.join(', ')}`);
  }
  
  if (failedPackages.length > 0) {
    console.error(`[WARNING] Failed to load: ${failedPackages.join(', ')}`);
  }
  
  return { 
    success: failedPackages.length === 0, 
    loadedPackages, 
    failedPackages 
  };
}

for await (const line of readLines(Deno.stdin)) {
  let input;
  try {
    input = JSON.parse(line);
  } catch (error) {
    console.log(JSON.stringify({
      error: "Invalid JSON input: " + error.message,
      errorType: "ValueError"
    }));
    continue;
  }

  if (input.mount_file) {
      const hostPath = input.mount_file;
      const virtualPath = input.virtual_path || hostPath;
      try {
          const contents = await Deno.readFile(hostPath);
          const dirs = virtualPath.split('/').slice(1, -1);
          let cur = '';
          for (const d of dirs) {
              cur += '/' + d;
              try {
                  pyodide.FS.mkdir(cur);
              } catch (e) {
                  if (!(e && e.message && e.message.includes('File exists'))) {
                      console.error("[DEBUG] Error creating directory in Pyodide file system:", cur, "|", e.message);
                  }
              }
          }
          pyodide.FS.writeFile(virtualPath, contents);
      } catch (e) {
          console.log(JSON.stringify({error: "Failed to mount file: " + e.message}));
      }
      continue;      
  }

  if (input.sync_file) {
      const virtualPath = input.sync_file;
      const hostPath = input.host_file || virtualPath;
      try {
          const contents = pyodide.FS.readFile(virtualPath);
          await Deno.writeFile(hostPath, contents);
      } catch (e) {
          console.error("[DEBUG] Failed to sync file:", hostPath, "|", e.message);
      }
      continue;
  }

  // Expecting an object like { "code": "...", ... }
  if (typeof input !== 'object' || input === null) {
    console.log(JSON.stringify({
      error: "Input is not a JSON object",
      errorType: "ValueError"
    }));
    continue;
  }

  // Check for shutdown
  if (input.shutdown) {
    break;
  }

  const code = input.code || "";

  try {
    // STEP 1: Load all required packages FIRST (following user's working pattern)
    const packageResult = await loadRequiredPackages(code);
    
    // STEP 2: Setup Python environment for code execution
    pyodide.runPython(`
import sys
import io
import json

def last_exception_args():
    return json.dumps(sys.last_exc.args) if sys.last_exc else None 

class FinalAnswer(Exception):
    pass

def final_answer(*args):
    raise FinalAnswer(*args)

# Keep references to the old stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr

# Create buffers for capturing output
execution_buffer = io.StringIO()
stderr_buffer = io.StringIO()

# Redirect stderr and stdout to buffers
sys.stderr = stderr_buffer
sys.stdout = execution_buffer
    `);

    // STEP 3: Execute the user code (packages should now be available)
    const result = await pyodide.runPythonAsync(code);

    // STEP 4: Get execution output and clean up
    const executionOutput = pyodide.runPython("execution_buffer.getvalue()");
    const stderrOutput = pyodide.runPython("stderr_buffer.getvalue()");

    // Restore stdout/stderr
    pyodide.runPython(`
sys.stdout = old_stdout
sys.stderr = old_stderr
    `);

    // STEP 5: Return the result
    let output;
    if (result !== null && result !== undefined) {
      output = result;
    } else if (executionOutput.trim()) {
      output = executionOutput;
    } else {
      output = null;
    }

    console.log(JSON.stringify({ output }));

  } catch (error) {
    // Handle errors gracefully
    const errorType = error.type || "Error";
    const errorMessage = (error.message || "").trim();
    let errorArgs = [];
    
    if (errorType !== "SyntaxError") {
      try {
        const last_exception_args = pyodide.globals.get("last_exception_args");
        errorArgs = JSON.parse(last_exception_args()) || [];
      } catch (e) {
        // Ignore errors getting exception args
      }
    }

    console.log(JSON.stringify({
      error: errorMessage,
      errorArgs: errorArgs,
      errorType: errorType
    }));
  }
} 