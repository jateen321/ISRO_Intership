// Vite's `?worker` import suffix (see vite/client.d.ts) resolves a module
// to a Worker constructor at build/transform time. Declared locally rather
// than pulling in the full `vite/client` ambient type set project-wide.
declare module '*?worker' {
  const WorkerConstructor: new () => Worker;
  export default WorkerConstructor;
}
