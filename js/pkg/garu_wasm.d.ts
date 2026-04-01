/* tslint:disable */
/* eslint-disable */

export class GaruWasm {
    free(): void;
    [Symbol.dispose](): void;
    analyze(text: string): any;
    analyze_topn(text: string, n: number): any;
    /**
     * Load CNN reranker model.
     */
    load_cnn(cnn_data: Uint8Array): void;
    constructor(model_data: Uint8Array);
    tokenize(text: string): any;
    static version(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_garuwasm_free: (a: number, b: number) => void;
    readonly garuwasm_analyze: (a: number, b: number, c: number) => [number, number, number];
    readonly garuwasm_analyze_topn: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly garuwasm_load_cnn: (a: number, b: number, c: number) => [number, number];
    readonly garuwasm_new: (a: number, b: number) => [number, number, number];
    readonly garuwasm_tokenize: (a: number, b: number, c: number) => [number, number, number];
    readonly garuwasm_version: () => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
