import ts from 'typescript';
import { ESLint } from 'eslint';
import { fileURLToPath } from 'node:url';
import { relative } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const configPath = fileURLToPath(new URL('../../tsconfig.m1010r.json', import.meta.url));
const config = ts.readConfigFile(configPath, ts.sys.readFile);
if (config.error) throw new Error(ts.flattenDiagnosticMessageText(config.error.messageText, '\n'));
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root);
const program = ts.createProgram(parsed.fileNames, parsed.options);
const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
if (diagnostics.length) {
  console.error(ts.formatDiagnosticsWithColorAndContext(diagnostics, {
    getCanonicalFileName: path => path, getCurrentDirectory: () => root, getNewLine: () => '\n',
  }));
  process.exitCode = 1;
} else {
  const eslint = new ESLint({ cwd: root, ignore: false, overrideConfig: [{
    files: ['tools/m1010/**/*.ts', 'tools/m1010r/**/*.ts'],
    languageOptions: { parserOptions: { projectService: false, project: configPath } },
  }] });
  const results = await eslint.lintFiles(parsed.fileNames);
  const errors = results.reduce((total, result) => total + result.errorCount, 0);
  const warnings = results.reduce((total, result) => total + result.warningCount, 0);
  console.log(await (await eslint.loadFormatter('stylish')).format(results));
  console.log(JSON.stringify({ files: parsed.fileNames.map(path => relative(root, path)), typecheck: 'PASS', errors, warnings }));
  if (errors || warnings) process.exitCode = 1;
}