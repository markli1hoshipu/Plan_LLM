/* ============================================================
 * Self-CriTeach — Blocks-World Demo
 * Self-bootstrapping SVG animation of a single-arm robot
 * solving PDDL blocksworld-style planning problems.
 *
 * Mounts into <div id="blocks-demo"></div>.
 * No external dependencies.
 * ============================================================ */

(function () {
  'use strict';

  // ----------------------------------------------------------
  // Demo problems  (cycled on loop)
  //
  // Plan steps are PDDL-style atomic actions. The gripper is "held"
  // between an acquire action and the matching release action.
  //
  //   { action: 'pick-up',  block, from: 'table' }      acquire from table
  //   { action: 'unstack',  block, from: <block-id> }   acquire from a block
  //   { action: 'put-down', block, to:   'table' }      release onto table
  //   { action: 'stack',    block, to:   <block-id> }   release onto a block
  // ----------------------------------------------------------
  var PROBLEMS = [
    {
      name: 'Problem 1',
      goal: '3 on 4; tower 8-5-7; 2 and 6 alone',
      blocks: {
        2: { color: '#ef4444' },  // red
        3: { color: '#f59e0b' },  // amber
        4: { color: '#10b981' },  // emerald
        5: { color: '#3b82f6' },  // blue
        6: { color: '#a855f7' },  // purple
        7: { color: '#ec4899' },  // pink
        8: { color: '#14b8a6' }   // teal
      },
      // initial layout: each entry is a stack, bottom-to-top, left-to-right.
      initial: [[4], [2], [5], [6], [8], [7, 3]],
      plan: [
        { action: 'unstack', block: 3, from: 7       },
        { action: 'stack',   block: 3, to:   4       },
        { action: 'pick-up', block: 5, from: 'table' },
        { action: 'stack',   block: 5, to:   8       },
        { action: 'pick-up', block: 7, from: 'table' },
        { action: 'stack',   block: 7, to:   5       }
      ]
    },
    {
      name: 'Problem 2',
      goal: 'stack all four blocks into one tower',
      blocks: {
        1: { color: '#3b82f6' },
        2: { color: '#10b981' },
        3: { color: '#f59e0b' },
        4: { color: '#ef4444' }
      },
      initial: [[2, 1], [4], [3]],
      plan: [
        { action: 'unstack', block: 1, from: 2       },
        { action: 'put-down', block: 1, to: 'table'  },
        { action: 'pick-up', block: 3, from: 'table' },
        { action: 'stack',   block: 3, to:   4       },
        { action: 'pick-up', block: 2, from: 'table' },
        { action: 'stack',   block: 2, to:   3       },
        { action: 'pick-up', block: 1, from: 'table' },
        { action: 'stack',   block: 1, to:   2       }
      ]
    },
    {
      name: 'Problem 3',
      goal: 'invert stack to 9 on 6 on 5',
      blocks: {
        5: { color: '#a855f7' },
        6: { color: '#14b8a6' },
        9: { color: '#ec4899' }
      },
      initial: [[5, 9], [6]],
      plan: [
        { action: 'unstack', block: 9, from: 5       },
        { action: 'put-down', block: 9, to: 'table'  },
        { action: 'pick-up', block: 6, from: 'table' },
        { action: 'stack',   block: 6, to:   5       },
        { action: 'pick-up', block: 9, from: 'table' },
        { action: 'stack',   block: 9, to:   6       }
      ]
    }
  ];

  // ----------------------------------------------------------
  // Visual constants
  // ----------------------------------------------------------
  var VB_W = 1600;          // viewBox width
  var VB_H = 900;           // viewBox height
  var BLOCK_W = 130;
  var BLOCK_H = 110;
  var BLOCK_GAP = 50;       // horizontal gap between adjacent columns
  var TABLE_Y = 720;        // top of the table surface (= bottom-y of any block on the table)
  var HOVER_Y = 180;        // y where gripper waits between actions
  var GRIPPER_TIP_OFFSET = 100; // gripper origin → claw tips (block top sits here when held)
  var SVGNS = 'http://www.w3.org/2000/svg';

  // ----------------------------------------------------------
  // Easing
  // ----------------------------------------------------------
  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // ----------------------------------------------------------
  // SVG helpers
  // ----------------------------------------------------------
  function svg(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) {
          el.setAttribute(k, attrs[k]);
        }
      }
    }
    return el;
  }

  function lighten(hex, amt) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    r = Math.min(255, Math.round(r + (255 - r) * amt));
    g = Math.min(255, Math.round(g + (255 - g) * amt));
    b = Math.min(255, Math.round(b + (255 - b) * amt));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  function darken(hex, amt) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    r = Math.max(0, Math.round(r * (1 - amt)));
    g = Math.max(0, Math.round(g * (1 - amt)));
    b = Math.max(0, Math.round(b * (1 - amt)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  // ----------------------------------------------------------
  // Renderer
  // ----------------------------------------------------------
  function BlocksDemo(container) {
    this.container = container;
    this.problemIndex = 0;
    this.stepIndex = 0;
    this.blocks = {};          // id -> { el, x, y, color }
    this.stacks = [];          // array of arrays of block ids (column slots, length fixed per problem)
    this.gripper = null;
    this.gripperState = { x: VB_W / 2, y: HOVER_Y, holding: null, open: 1 };
    this.running = false;
    this.reducedMotion = false;
    this._raf = null;
    this._build();
  }

  BlocksDemo.prototype._build = function () {
    var c = this.container;
    c.innerHTML = '';

    // --- overlay (chip + title) ---
    this.overlay = document.createElement('div');
    this.overlay.className = 'bw-overlay';

    var title = document.createElement('div');
    title.className = 'bw-title';
    var dot = document.createElement('span');
    dot.className = 'bw-title-dot';
    title.appendChild(dot);
    title.appendChild(document.createTextNode('Blocks-World Planner'));
    this.overlay.appendChild(title);

    // right-aligned vertical column: action chip on top, goal beneath it
    this.topRight = document.createElement('div');
    this.topRight.className = 'bw-top-right';

    this.chip = document.createElement('div');
    this.chip.className = 'bw-chip bw-hidden';
    this.chip.innerHTML = '<span class="bw-kw">init</span>';
    this.topRight.appendChild(this.chip);

    this.goalEl = document.createElement('div');
    this.goalEl.className = 'bw-goal';
    this.topRight.appendChild(this.goalEl);

    this.overlay.appendChild(this.topRight);

    // --- footer (progress only) ---
    this.footer = document.createElement('div');
    this.footer.className = 'bw-footer';

    this.progressEl = document.createElement('div');
    this.progressEl.className = 'bw-progress';
    this.footer.appendChild(this.progressEl);

    // --- thinking pill ---
    this.thinkingEl = document.createElement('div');
    this.thinkingEl.className = 'bw-thinking';
    this.thinkingEl.innerHTML =
      '<div class="bw-thinking-pill">' +
      '<span>Planning next problem</span>' +
      '<span class="bw-thinking-dots"><span></span><span></span><span></span></span>' +
      '</div>';

    // --- stage / SVG ---
    var stage = document.createElement('div');
    stage.className = 'bw-stage';
    stage.setAttribute('role', 'img');
    stage.setAttribute('aria-label',
      'Animated demo of a robot arm solving block-stacking planning problems.');

    var s = svg('svg', {
      viewBox: '0 0 ' + VB_W + ' ' + VB_H,
      preserveAspectRatio: 'xMidYMid meet'
    });
    this.svg = s;

    var defs = svg('defs');

    // soft drop-shadow for blocks
    var f = svg('filter', { id: 'bw-shadow', x: '-20%', y: '-20%', width: '140%', height: '140%' });
    var fb = svg('feGaussianBlur', { in: 'SourceAlpha', stdDeviation: '6' });
    var fo = svg('feOffset', { dx: '0', dy: '6', result: 'offb' });
    var fc = svg('feComponentTransfer');
    var ft = svg('feFuncA', { type: 'linear', slope: '0.35' });
    fc.appendChild(ft);
    var fm = svg('feMerge');
    fm.appendChild(svg('feMergeNode'));
    fm.appendChild(svg('feMergeNode', { in: 'SourceGraphic' }));
    f.appendChild(fb); f.appendChild(fo); f.appendChild(fc); f.appendChild(fm);
    defs.appendChild(f);

    // table gradient
    var tg = svg('linearGradient', { id: 'bw-table', x1: '0', y1: '0', x2: '0', y2: '1' });
    tg.appendChild(svg('stop', { offset: '0%',  'stop-color': '#cbd5e0' }));
    tg.appendChild(svg('stop', { offset: '100%', 'stop-color': '#e2e8f0' }));
    defs.appendChild(tg);

    // grid lines pattern
    var pat = svg('pattern', {
      id: 'bw-grid', width: '80', height: '80',
      patternUnits: 'userSpaceOnUse'
    });
    pat.appendChild(svg('path', {
      d: 'M 80 0 L 0 0 0 80',
      fill: 'none',
      stroke: 'rgba(49,130,206,0.05)',
      'stroke-width': '1'
    }));
    defs.appendChild(pat);

    s.appendChild(defs);

    // backdrop grid
    s.appendChild(svg('rect', {
      x: '0', y: '0', width: VB_W, height: VB_H, fill: 'url(#bw-grid)'
    }));

    // table top
    s.appendChild(svg('rect', {
      x: '40', y: TABLE_Y, width: VB_W - 80, height: 22, rx: '6',
      fill: 'url(#bw-table)'
    }));
    // table edge shadow
    s.appendChild(svg('rect', {
      x: '40', y: TABLE_Y + 22, width: VB_W - 80, height: 5, rx: '2',
      fill: 'rgba(15,23,42,0.10)'
    }));

    // layers
    this.blockLayer = svg('g'); s.appendChild(this.blockLayer);
    this.gripperLayer = svg('g'); s.appendChild(this.gripperLayer);

    this._buildGripper();

    stage.appendChild(s);
    stage.appendChild(this.overlay);
    stage.appendChild(this.footer);
    stage.appendChild(this.thinkingEl);
    c.appendChild(stage);
  };

  BlocksDemo.prototype._buildGripper = function () {
    var g = svg('g', { id: 'bw-gripper' });

    // vertical rail (so it looks anchored to the ceiling)
    this.rail = svg('rect', {
      x: '-6', y: '0', width: '12', height: HOVER_Y - 20,
      fill: '#cbd5e0', rx: '4'
    });
    g.appendChild(this.rail);

    // arm body
    g.appendChild(svg('rect', {
      x: '-50', y: '-40', width: '100', height: '60', rx: '12',
      fill: '#1a365d', stroke: '#0f172a', 'stroke-width': '2'
    }));
    g.appendChild(svg('rect', {
      x: '-46', y: '-36', width: '92', height: '12', rx: '6',
      fill: 'rgba(255,255,255,0.18)'
    }));
    g.appendChild(svg('rect', {
      x: '-50', y: '-10', width: '100', height: '4',
      fill: '#3182ce'
    }));

    // pincers (claws)
    this.leftPincer = svg('g', { id: 'bw-left-pincer' });
    this.leftPincer.appendChild(svg('rect', {
      x: '-58', y: '20', width: '14', height: '60', rx: '4',
      fill: '#2d3748'
    }));
    this.leftPincer.appendChild(svg('polygon', {
      points: '-58,80 -44,80 -52,98',
      fill: '#1a202c'
    }));
    g.appendChild(this.leftPincer);

    this.rightPincer = svg('g', { id: 'bw-right-pincer' });
    this.rightPincer.appendChild(svg('rect', {
      x: '44', y: '20', width: '14', height: '60', rx: '4',
      fill: '#2d3748'
    }));
    this.rightPincer.appendChild(svg('polygon', {
      points: '44,80 58,80 52,98',
      fill: '#1a202c'
    }));
    g.appendChild(this.rightPincer);

    this.gripper = g;
    this.gripperLayer.appendChild(g);
    this._updateGripperTransform();
  };

  BlocksDemo.prototype._updateGripperTransform = function () {
    var s = this.gripperState;
    this.gripper.setAttribute('transform',
      'translate(' + s.x + ',' + s.y + ')');
    // Rail runs from top of stage to just above the gripper body.
    this.rail.setAttribute('y', String(-s.y));
    this.rail.setAttribute('height', String(Math.max(0, s.y - 40)));

    // open factor 0..1: 0 closed, 1 open  → shift pincers outward
    var dx = 10 * s.open;
    this.leftPincer.setAttribute('transform',
      'translate(' + (-dx) + ',0)');
    this.rightPincer.setAttribute('transform',
      'translate(' + (dx) + ',0)');
  };

  // ------------------------------------------------------------
  // Block factory — gradient + shadow + label
  // ------------------------------------------------------------
  BlocksDemo.prototype._makeBlock = function (id, color) {
    var gradId = 'bw-bg-' + id;
    var defs = this.svg.querySelector('defs');
    var prior = defs.querySelector('#' + gradId);
    if (prior) defs.removeChild(prior);

    var grad = svg('linearGradient', {
      id: gradId, x1: '0', y1: '0', x2: '0', y2: '1'
    });
    grad.appendChild(svg('stop', { offset: '0%',  'stop-color': lighten(color, 0.35) }));
    grad.appendChild(svg('stop', { offset: '55%', 'stop-color': color }));
    grad.appendChild(svg('stop', { offset: '100%', 'stop-color': darken(color, 0.15) }));
    defs.appendChild(grad);

    var g = svg('g', { class: 'bw-block', 'data-id': id });

    g.appendChild(svg('rect', {
      x: -BLOCK_W / 2, y: -BLOCK_H,
      width: BLOCK_W, height: BLOCK_H, rx: '12',
      fill: 'url(#' + gradId + ')',
      stroke: darken(color, 0.25),
      'stroke-width': '2',
      filter: 'url(#bw-shadow)'
    }));

    g.appendChild(svg('rect', {
      x: -BLOCK_W / 2 + 8, y: -BLOCK_H + 8,
      width: BLOCK_W - 16, height: 14, rx: '7',
      fill: 'rgba(255,255,255,0.4)'
    }));
    g.appendChild(svg('rect', {
      x: -BLOCK_W / 2 + 6, y: -16,
      width: BLOCK_W - 12, height: 6, rx: '3',
      fill: 'rgba(0,0,0,0.12)'
    }));

    var label = svg('text', {
      x: '0', y: -BLOCK_H / 2 + 12,
      'text-anchor': 'middle',
      'font-family': "'Inter','Source Sans Pro',sans-serif",
      'font-weight': '700',
      'font-size': '46',
      fill: 'white',
      'paint-order': 'stroke',
      stroke: 'rgba(0,0,0,0.18)',
      'stroke-width': '3'
    });
    label.textContent = String(id);
    g.appendChild(label);

    return g;
  };

  // ------------------------------------------------------------
  // Layout: compute x for each column slot. Column count is fixed
  // for the duration of a problem (preallocated in _loadProblem),
  // so column x-positions never shift between steps.
  // ------------------------------------------------------------
  BlocksDemo.prototype._columnX = function (col) {
    var n = this.stacks.length;
    var totalW = n * BLOCK_W + (n - 1) * BLOCK_GAP;
    var startX = (VB_W - totalW) / 2 + BLOCK_W / 2;
    return startX + col * (BLOCK_W + BLOCK_GAP);
  };

  // y position (bottom-y) for the block at depth `depth` in a stack
  // (depth 0 = bottom, sitting on the table).
  BlocksDemo.prototype._depthY = function (depth) {
    return TABLE_Y - depth * BLOCK_H;
  };

  // ------------------------------------------------------------
  // Lay out every block on its column according to this.stacks.
  // Animates if duration > 0, else snaps immediately.
  // The held block (if any) is left alone.
  // ------------------------------------------------------------
  BlocksDemo.prototype._relayoutAll = function (duration) {
    var self = this;
    var targets = {};
    for (var col = 0; col < this.stacks.length; col++) {
      var stack = this.stacks[col];
      var x = this._columnX(col);
      for (var depth = 0; depth < stack.length; depth++) {
        var id = stack[depth];
        targets[id] = { x: x, y: this._depthY(depth) };
      }
    }
    // snap held block stays in current position
    var promises = [];
    Object.keys(this.blocks).forEach(function (id) {
      if (id === String(self.gripperState.holding)) return;
      var t = targets[id];
      var b = self.blocks[id];
      if (!t) return;
      if (b.x === t.x && b.y === t.y) return;
      if (!duration) {
        b.x = t.x; b.y = t.y;
        b.el.setAttribute('transform', 'translate(' + t.x + ',' + t.y + ')');
        return;
      }
      var fromX = b.x, fromY = b.y;
      promises.push(self._tween(duration, easeInOutCubic, function (e) {
        b.x = fromX + (t.x - fromX) * e;
        b.y = fromY + (t.y - fromY) * e;
        b.el.setAttribute('transform', 'translate(' + b.x + ',' + b.y + ')');
      }));
    });
    return Promise.all(promises);
  };

  BlocksDemo.prototype._loadProblem = function (idx) {
    var prob = PROBLEMS[idx];
    this.problemIndex = idx;
    this.stepIndex = 0;

    // clear blocks
    this.blockLayer.innerHTML = '';
    this.blocks = {};

    // Preallocate columns: one column per block. The initial stacks fill
    // the first columns; remaining columns start empty so subsequent
    // table placements never have to grow the layout. This keeps every
    // existing block visually pinned to a stable column.
    var totalBlocks = Object.keys(prob.blocks).length;
    var initial = prob.initial.map(function (s) { return s.slice(); });
    while (initial.length < totalBlocks) initial.push([]);
    this.stacks = initial;

    // create block elements at their initial positions
    for (var col = 0; col < this.stacks.length; col++) {
      var stack = this.stacks[col];
      var x = this._columnX(col);
      for (var depth = 0; depth < stack.length; depth++) {
        var id = stack[depth];
        var color = prob.blocks[id].color;
        var b = this._makeBlock(id, color);
        this.blockLayer.appendChild(b);
        var y = this._depthY(depth);
        this.blocks[id] = { el: b, x: x, y: y, color: color };
        b.setAttribute('transform', 'translate(' + x + ',' + y + ')');
      }
    }

    // gripper starting position: centered, holding nothing
    this.gripperState.x = VB_W / 2;
    this.gripperState.y = HOVER_Y;
    this.gripperState.holding = null;
    this.gripperState.open = 1;
    this._updateGripperTransform();

    this.goalEl.textContent = 'Goal: ' + prob.goal;

    // progress dots
    this.progressEl.innerHTML = '';
    for (var i = 0; i < prob.plan.length; i++) {
      var d = document.createElement('div');
      d.className = 'bw-progress-dot';
      this.progressEl.appendChild(d);
    }
  };

  // ------------------------------------------------------------
  // Tween helpers
  // ------------------------------------------------------------
  BlocksDemo.prototype._tween = function (duration, easing, onUpdate) {
    var self = this;
    if (this.reducedMotion) {
      onUpdate(1);
      return Promise.resolve();
    }
    return new Promise(function (resolve) {
      var start = performance.now();
      function step(now) {
        var t = Math.min(1, (now - start) / duration);
        var e = easing(t);
        onUpdate(e);
        if (t < 1) {
          self._raf = requestAnimationFrame(step);
        } else {
          resolve();
        }
      }
      self._raf = requestAnimationFrame(step);
    });
  };

  BlocksDemo.prototype._wait = function (ms) {
    if (this.reducedMotion) return Promise.resolve();
    return new Promise(function (r) { setTimeout(r, ms); });
  };

  // Move gripper from current state to (x, y); held block follows.
  BlocksDemo.prototype._moveGripper = function (toX, toY, duration) {
    var self = this;
    var s = self.gripperState;
    var fromX = s.x, fromY = s.y;
    var held = s.holding;
    return this._tween(duration, easeInOutCubic, function (t) {
      s.x = fromX + (toX - fromX) * t;
      s.y = fromY + (toY - fromY) * t;
      self._updateGripperTransform();
      if (held != null && self.blocks[held]) {
        // block's bottom-y sits one BLOCK_H below the gripper claws top,
        // i.e. claws span the height of the block: bottom-y = gripper.y + TIP_OFFSET
        var bx = s.x;
        var by = s.y + GRIPPER_TIP_OFFSET;
        self.blocks[held].x = bx;
        self.blocks[held].y = by;
        self.blocks[held].el.setAttribute('transform',
          'translate(' + bx + ',' + by + ')');
      }
    });
  };

  // Open / close pincers (open: 0 closed, 1 open)
  BlocksDemo.prototype._setOpen = function (target, duration) {
    var self = this;
    var s = self.gripperState;
    var from = s.open;
    return this._tween(duration, easeInOutCubic, function (t) {
      s.open = from + (target - from) * t;
      self._updateGripperTransform();
    });
  };

  // ------------------------------------------------------------
  // State queries
  // ------------------------------------------------------------
  BlocksDemo.prototype._findStack = function (blockId) {
    for (var i = 0; i < this.stacks.length; i++) {
      var idx = this.stacks[i].indexOf(blockId);
      if (idx !== -1) return { stack: i, idx: idx };
    }
    return null;
  };

  // First empty column (left-to-right). Always exists because we
  // preallocate one column per block.
  BlocksDemo.prototype._firstEmptyColumn = function () {
    for (var i = 0; i < this.stacks.length; i++) {
      if (this.stacks[i].length === 0) return i;
    }
    return -1;
  };

  // Raise a block element to render on top of all other blocks.
  BlocksDemo.prototype._raiseBlock = function (id) {
    var b = this.blocks[id];
    if (!b) return;
    this.blockLayer.appendChild(b.el); // moves to end → drawn last → on top
  };

  // ------------------------------------------------------------
  // UI updates
  // ------------------------------------------------------------
  BlocksDemo.prototype._setChip = function (step) {
    if (!step) {
      this.chip.classList.add('bw-hidden');
      return;
    }
    if (step.action === 'goal') {
      this.chip.innerHTML = '<span class="bw-kw">goal</span> reached ✓';
      this.chip.classList.remove('bw-hidden');
      return;
    }
    // Render as PDDL-like: action(block, source-or-dest, robot)
    var arg = (step.action === 'pick-up' || step.action === 'unstack')
      ? step.from : step.to;
    var html = '<span class="bw-kw">' + step.action + '</span>('
             + step.block + ', ' + arg + ', r1)';
    this.chip.innerHTML = html;
    this.chip.classList.remove('bw-hidden');
  };

  BlocksDemo.prototype._markProgress = function (idx, state) {
    var dots = this.progressEl.querySelectorAll('.bw-progress-dot');
    for (var i = 0; i < dots.length; i++) {
      dots[i].classList.remove('bw-active', 'bw-done');
      if (i < idx) dots[i].classList.add('bw-done');
      else if (i === idx && state === 'active') dots[i].classList.add('bw-active');
      else if (i === idx && state === 'done') dots[i].classList.add('bw-done');
    }
  };

  // ------------------------------------------------------------
  // Atomic action implementations
  //
  // Each step is one PDDL action. The gripper holds the block between
  // an acquire-action (pick-up/unstack) and a release-action (stack/
  // put-down). We do NOT do pick-and-place in one step.
  // ------------------------------------------------------------

  // Acquire: pick-up (from table) or unstack (from another block).
  // Preconditions assumed valid: block is on top of its stack and
  // the gripper is empty.
  BlocksDemo.prototype._actionAcquire = function (step) {
    var self = this;
    var block = step.block;
    var bData = this.blocks[block];
    if (!bData) return Promise.resolve();

    // Raise so it draws on top during the descent / lift.
    this._raiseBlock(block);

    // 1) Travel above the block at hover height
    return this._moveGripper(bData.x, HOVER_Y, 600)
      // 2) Open pincers (in case partially closed)
      .then(function () { return self._setOpen(1, 180); })
      // 3) Descend so the claws straddle the block
      .then(function () {
        // gripper.y such that claws bottom (gripper.y + GRIPPER_TIP_OFFSET)
        // reaches the bottom of the block (bData.y). i.e. gripper.y =
        // bData.y - GRIPPER_TIP_OFFSET. Slight lift (-6) for a small gap.
        var targetY = bData.y - GRIPPER_TIP_OFFSET - 6;
        return self._moveGripper(bData.x, targetY, 480);
      })
      // 4) Close pincers
      .then(function () { return self._setOpen(0.25, 240); })
      // 5) Update model: block is now held, remove from its source stack
      .then(function () {
        self.gripperState.holding = block;
        var loc = self._findStack(block);
        if (loc) self.stacks[loc.stack].splice(loc.idx, 1);
        return self._wait(80);
      })
      // 6) Lift to hover height
      .then(function () { return self._moveGripper(self.gripperState.x, HOVER_Y, 480); });
  };

  // Release: stack (onto another block) or put-down (onto table).
  // Preconditions assumed valid: gripper is holding `step.block`.
  BlocksDemo.prototype._actionRelease = function (step) {
    var self = this;
    var block = step.block;
    var bData = this.blocks[block];
    if (!bData) return Promise.resolve();

    // Resolve destination column + final bottom-y
    var destCol, destY;
    if (step.to === 'table') {
      destCol = this._firstEmptyColumn();
      if (destCol === -1) {
        // shouldn't happen since columns are preallocated, but be safe
        this.stacks.push([]);
        destCol = this.stacks.length - 1;
      }
      destY = TABLE_Y;
    } else {
      var loc = this._findStack(step.to);
      if (!loc) return Promise.resolve();
      destCol = loc.stack;
      destY = this._depthY(this.stacks[destCol].length); // top of stack
    }
    var destX = this._columnX(destCol);

    // 1) Lateral travel to destination at hover height
    return this._moveGripper(destX, HOVER_Y, 700)
      // 2) Descend to placement
      .then(function () {
        var targetY = destY - GRIPPER_TIP_OFFSET - 6;
        return self._moveGripper(destX, targetY, 480);
      })
      // 3) Open pincers — block "lands"
      .then(function () { return self._setOpen(1, 240); })
      // 4) Commit to model + snap block to canonical resting position
      .then(function () {
        self.stacks[destCol].push(block);
        bData.x = destX;
        bData.y = destY;
        bData.el.setAttribute('transform',
          'translate(' + destX + ',' + destY + ')');
        self.gripperState.holding = null;
        // 5) Lift the gripper away
        return self._moveGripper(destX, HOVER_Y, 450);
      });
  };

  BlocksDemo.prototype._executeStep = function (step) {
    if (step.action === 'pick-up' || step.action === 'unstack') {
      return this._actionAcquire(step);
    }
    if (step.action === 'stack' || step.action === 'put-down') {
      return this._actionRelease(step);
    }
    return Promise.resolve();
  };

  BlocksDemo.prototype._runProblem = function () {
    var self = this;
    var prob = PROBLEMS[self.problemIndex];

    function nextStep() {
      if (!self.running) return Promise.resolve();
      if (self.stepIndex >= prob.plan.length) {
        // mark all done
        var dots = self.progressEl.querySelectorAll('.bw-progress-dot');
        for (var i = 0; i < dots.length; i++) {
          dots[i].classList.remove('bw-active');
          dots[i].classList.add('bw-done');
        }
        self._setChip({ action: 'goal' });
        return self._wait(1400);
      }
      var step = prob.plan[self.stepIndex];
      self._setChip(step);
      self._markProgress(self.stepIndex, 'active');
      return self._executeStep(step).then(function () {
        self.stepIndex++;
        return nextStep();
      });
    }

    return nextStep();
  };

  BlocksDemo.prototype._showThinking = function (show) {
    if (show) this.thinkingEl.classList.add('bw-visible');
    else this.thinkingEl.classList.remove('bw-visible');
  };

  // Run the full loop, cycling through problems
  BlocksDemo.prototype.start = function () {
    var self = this;
    if (self.running) return;
    self.running = true;

    function loop() {
      if (!self.running) return;
      self._loadProblem(self.problemIndex);
      self._setChip(null);
      self._wait(600).then(function () {
        return self._runProblem();
      }).then(function () {
        if (!self.running) return;
        self._showThinking(true);
        self._setChip(null);
        return self._wait(1300).then(function () {
          self._showThinking(false);
          self.problemIndex = (self.problemIndex + 1) % PROBLEMS.length;
          return self._wait(250);
        });
      }).then(function () {
        if (self.running) loop();
      });
    }

    loop();
  };

  BlocksDemo.prototype.stop = function () {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
  };

  // Static final state for reduced-motion users: simulate the plan
  // for the current problem instantly and render the resulting layout.
  BlocksDemo.prototype.staticFinalState = function () {
    this.reducedMotion = true;
    var idx = this.problemIndex || 0;
    this._loadProblem(idx);
    var prob = PROBLEMS[idx];

    // Run the plan synchronously through the state model only.
    for (var i = 0; i < prob.plan.length; i++) {
      var step = prob.plan[i];
      if (step.action === 'pick-up' || step.action === 'unstack') {
        var loc = this._findStack(step.block);
        if (loc) this.stacks[loc.stack].splice(loc.idx, 1);
        this.gripperState.holding = step.block;
      } else if (step.action === 'stack') {
        var dloc = this._findStack(step.to);
        if (dloc) this.stacks[dloc.stack].push(step.block);
        this.gripperState.holding = null;
      } else if (step.action === 'put-down') {
        var ec = this._firstEmptyColumn();
        if (ec === -1) { this.stacks.push([]); ec = this.stacks.length - 1; }
        this.stacks[ec].push(step.block);
        this.gripperState.holding = null;
      }
    }

    // Snap every block to its computed final position.
    this._relayoutAll(0);

    // park gripper to the side, closed
    this.gripperState.x = VB_W - 220;
    this.gripperState.y = HOVER_Y;
    this.gripperState.open = 1;
    this.gripperState.holding = null;
    this._updateGripperTransform();

    this.goalEl.textContent = 'Goal reached: ' + prob.goal;
    this.progressEl.innerHTML = '';
    for (var j = 0; j < prob.plan.length; j++) {
      var d = document.createElement('div');
      d.className = 'bw-progress-dot bw-done';
      this.progressEl.appendChild(d);
    }
    this.chip.classList.remove('bw-hidden');
    this.chip.innerHTML = '[<span class="bw-kw">goal</span>, reached ✓]';
  };

  // Pause / resume on visibility change to save CPU
  BlocksDemo.prototype._installVisibilityHooks = function () {
    var self = this;
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) self.stop();
      else if (!self.reducedMotion) self.start();
    });

    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !self.reducedMotion) self.start();
          else if (!e.isIntersecting) self.stop();
        });
      }, { threshold: 0.05 });
      io.observe(self.container);
    }
  };

  // ----------------------------------------------------------
  // Bootstrap
  // ----------------------------------------------------------
  function boot() {
    var el = document.getElementById('blocks-demo');
    if (!el) return;
    if (el.__bw_initialized) return;
    el.__bw_initialized = true;

    var demo = new BlocksDemo(el);

    var prefersReduced = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReduced) {
      demo.staticFinalState();
    } else {
      demo._loadProblem(0);
      demo.start();
      demo._installVisibilityHooks();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
