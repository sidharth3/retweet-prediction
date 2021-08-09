import Vue from 'vue';
declare type LocalData = {
    check: any;
    isFixed: boolean;
    lastScrollTop: Number;
};
declare const _default: import("vue/types/vue").ExtendedVue<Vue, LocalData, {
    getScrollTop(): number;
    main(): void;
    registerEvent(): void;
    removeEvent(): void;
}, unknown, {
    threshold: number;
    headerClass: string;
    fixedClass: string;
    hideScrollUp: boolean;
}>;
export default _default;
