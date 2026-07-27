/* tslint:disable */
/* eslint-disable */

export class GaruWasm {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * 사용자 단어 등록 — 모델 리빌드 없이 도메인 어휘를 인식시킨다.
     */
    add_user_word(surface: string, pos: string, freq?: number | null): void;
    analyze(text: string): any;
    analyze_topn(text: string, n: number): any;
    /**
     * 등록된 사용자 단어를 모두 제거.
     */
    clear_user_words(): void;
    constructor(model_data: Uint8Array, normalize_jamo?: boolean | null);
    tokenize(text: string): any;
    /**
     * 등록된 사용자 단어 수.
     */
    user_word_count(): number;
    static version(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_garuwasm_free: (a: number, b: number) => void;
    readonly garuwasm_add_user_word: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly garuwasm_analyze: (a: number, b: number, c: number, d: number) => void;
    readonly garuwasm_analyze_topn: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly garuwasm_clear_user_words: (a: number) => void;
    readonly garuwasm_new: (a: number, b: number, c: number, d: number) => void;
    readonly garuwasm_tokenize: (a: number, b: number, c: number, d: number) => void;
    readonly garuwasm_user_word_count: (a: number) => number;
    readonly garuwasm_version: (a: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
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
