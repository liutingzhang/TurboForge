// Uses the project's Vite/React build toolchain; no extra bundler dependency.
import {build} from '../web/node_modules/vite/dist/node/index.js';
import react from '../web/node_modules/@vitejs/plugin-react/dist/index.js';
import {mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const root = fileURLToPath(new URL('../',import.meta.url));
const output = fileURLToPath(new URL('../reports/assistant-browser/',import.meta.url));
await mkdir(output,{recursive:true});
await build({root:root+'web',configFile:false,plugins:[react()],
  resolve:{alias:Object.fromEntries(['react','react-dom','antd','@ant-design/icons'].map(name=>[name,root+'web/node_modules/'+name]))},
  define:{'process.env.NODE_ENV':'"production"'},
  build:{outDir:output,emptyOutDir:false,minify:true,cssCodeSplit:false,
    lib:{entry:root+'tests/browser/assistant-fixture.tsx',formats:['es'],fileName:()=> 'fixture.js',cssFileName:'fixture'}}});
await writeFile(output+'index.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>助手 DOM 验收</title><link rel="stylesheet" href="/__assistant_fixture/fixture.css"><div id="root"></div><script type="module" src="/__assistant_fixture/fixture.js"></script></html>');
console.log('Assistant browser fixture built; contains synthetic data only.');
