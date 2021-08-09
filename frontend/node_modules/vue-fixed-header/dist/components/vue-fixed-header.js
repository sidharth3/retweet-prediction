"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var vue_1 = tslib_1.__importDefault(require("vue"));
exports.default = vue_1.default.extend({
    props: {
        threshold: {
            required: false,
            type: Number,
            default: 0
        },
        headerClass: {
            required: false,
            type: String,
            default: 'vue-fixed-header'
        },
        fixedClass: {
            required: false,
            type: String,
            default: 'vue-fixed-header--isFixed'
        },
        hideScrollUp: {
            required: false,
            type: Boolean,
            default: false
        }
    },
    data: function () {
        return { check: null, isFixed: false, lastScrollTop: 0 };
    },
    mounted: function () {
        this.main();
        this.registerEvent();
    },
    beforeDestroy: function () {
        this.removeEvent();
    },
    methods: {
        getScrollTop: function () {
            return window.pageYOffset || document.documentElement.scrollTop;
        },
        main: function () {
            var _this = this;
            this.lastScrollTop = this.getScrollTop();
            this.check = function () {
                var _a = _this, threshold = _a.threshold, hideScrollUp = _a.hideScrollUp;
                var currentScrollPos = _this.getScrollTop();
                var isOverThreshold = currentScrollPos > threshold;
                var isScrollDown = currentScrollPos >= _this.lastScrollTop;
                var newIsFixed = isScrollDown
                    // down-scrolled - are we over threshold ?
                    ? isOverThreshold
                    // up-scrolled - EITHER we don't care OR are we over threshold ?
                    : (hideScrollUp ? false : isOverThreshold);
                _this.lastScrollTop = currentScrollPos;
                if (_this.isFixed !== newIsFixed) {
                    _this.isFixed = newIsFixed;
                    _this.$emit('change', _this.isFixed);
                }
            };
        },
        registerEvent: function () {
            window.addEventListener('scroll', this.check);
        },
        removeEvent: function () {
            window.removeEventListener('scroll', this.check);
        }
    },
    render: function (h) {
        var _a;
        var children = this.$slots.default;
        if (!children || !tslib_1.__spreadArrays(children)[0]) {
            return h();
        }
        var _child = tslib_1.__spreadArrays(children)[0];
        var child = h(_child.tag, _child.data, _child.children || _child.text);
        child.data = child.data || { class: '' };
        if (typeof child.data.class === 'string') {
            child.data.class = child.data.class.split(' ');
        }
        if (Array.isArray(child.data.class)) {
            child.data.class = tslib_1.__spreadArrays(child.data.class).reduce(function (a, b) {
                var _a;
                return (tslib_1.__assign(tslib_1.__assign({}, a), (_a = {}, _a[b] = true, _a)));
            }, {});
        }
        child.data.class = tslib_1.__assign(tslib_1.__assign({}, child.data.class), (_a = {}, _a[this.headerClass] = true, _a[this.fixedClass] = !!this.isFixed, _a));
        child.data.class = Object.entries(child.data.class)
            .map(function (_a) {
            var k = _a[0], v = _a[1];
            return (v ? k : null);
        })
            .filter(function (v) { return v; })
            .join(' ');
        return child;
    }
});
//# sourceMappingURL=vue-fixed-header.js.map