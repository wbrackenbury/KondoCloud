/**
 * Objects array of jQuery.Deferred that calls before elFinder boot up
 * 
 * @type Array
 */
dfrdsBeforeBootup = [],

/**
 * Plugin name to check for conflicts with bootstrap etc
 *
 * @type Array
 **/
conflictChecks = ['button', 'tooltip'],

/**
 * Node on which elfinder creating
 *
 * @type jQuery
 **/
node = $(elm),

/**
 * Object of events originally registered in this node
 * 
 * @type Object
 */
prevEvents = $.extend(true, {}, $._data(node.get(0), 'events')),

/**
 * Store node contents.
 *
 * @see this.destroy
 * @type jQuery
 **/
prevContent = $('<div/>').append(node.contents()).attr('class', node.attr('class') || '').attr('style', node.attr('style') || ''),

/**
 * Instance ID. Required to get/set cookie
 *
 * @type String
 **/
id = node.attr('id') || node.attr('id', 'elfauto' + $('.elfinder').length).attr('id'),

/**
 * Events namespace
 *
 * @type String
 **/
namespace = 'elfinder-' + id,

/**
 * Mousedown event
 *
 * @type String
 **/
mousedown = 'mousedown.'+namespace,

/**
 * Keydown event
 *
 * @type String
 **/
keydown = 'keydown.'+namespace,

/**
 * Keypress event
 *
 * @type String
 **/
keypress = 'keypress.'+namespace,

/**
 * Keypup event
 *
 * @type String
 **/
keyup    = 'keyup.'+namespace,

/**
 * Is shortcuts/commands enabled
 *
 * @type Boolean
 **/
enabled = false,

/**
 * Store enabled value before ajax request
 *
 * @type Boolean
 **/
prevEnabled = false,

/**
 * List of build-in events which mapped into methods with same names
 *
 * @type Array
 **/
events = ['enable', 'disable', 'load', 'open', 'reload', 'select',  'add', 'remove', 'change', 'dblclick', 'getfile', 'lockfiles', 'unlockfiles', 'selectfiles', 'unselectfiles', 'dragstart', 'dragstop', 'search', 'searchend', 'viewchange'],

/**
 * Rules to validate data from backend
 *
 * @type Object
 **/
rules = {},

/**
 * Current working directory hash
 *
 * @type String
 **/
cwd = '',

/**
 * Current working directory options default
 *
 * @type Object
 **/
cwdOptionsDefault = {
    path          : '',
    url           : '',
    tmbUrl        : '',
    disabled      : [],
    separator     : '/',
    archives      : [],
    extract       : [],
    copyOverwrite : true,
    uploadOverwrite : true,
    uploadMaxSize : 0,
    jpgQuality    : 100,
    tmbCrop       : false,
    tmb           : false // old API
},

/**
 * Current working directory options
 *
 * @type Object
 **/
cwdOptions = {},

/**
 * Files/dirs cache
 *
 * @type Object
 **/
files = {},

/**
 * Hidden Files/dirs cache
 *
 * @type Object
 **/
hiddenFiles = {},

/**
 * Files/dirs hash cache of each dirs
 *
 * @type Object
 **/
ownFiles = {},

/**
 * Selected files hashes
 *
 * @type Array
 **/
selected = [],

/**
 * Events listeners
 *
 * @type Object
 **/
listeners = {},

/**
 * Shortcuts
 *
 * @type Object
 **/
shortcuts = {},

/**
 * Buffer for copied files
 *
 * @type Array
 **/
clipboard = [],

/**
 * Copied/cuted files hashes
 * Prevent from remove its from cache.
 * Required for dispaly correct files names in error messages
 *
 * @type Object
 **/
remember = {},

/**
 * Queue for 'open' requests
 *
 * @type Array
 **/
queue = [],

/**
 * Queue for only cwd requests e.g. `tmb`
 *
 * @type Array
 **/
cwdQueue = [],

/**
 * Commands prototype
 *
 * @type Object
 **/
base = new self.command(self),

/**
 * elFinder node width
 *
 * @type String
 * @default "auto"
 **/
width  = 'auto',

/**
 * elFinder node height
 * Number: pixcel or String: Number + "%"
 *
 * @type Number | String
 * @default 400
 **/
height = 400,

/**
 * Base node object or selector
 * Element which is the reference of the height percentage
 *
 * @type Object|String
 * @default null | $(window) (if height is percentage)
 **/
heightBase = null,

/**
 * MIME type list(Associative array) handled as a text file
 * 
 * @type Object|null
 */
textMimes = null,

/**
 * elfinder path for sound played on remove
 * @type String
 * @default ./sounds/
 **/
soundPath = 'sounds/',

/**
 * JSON.stringify of previous fm.sorters
 * @type String
 */
prevSorterStr = '',

/**
 * Map table of file extention to MIME-Type
 * @type Object
 */
extToMimeTable,

/**
 * Disabled page unload function
 * @type Boolean
 */
diableUnloadCheck = false,













/**
 * Maximum number of concurrent connections on request
 * 
 * @type Number
 */
requestMaxConn,

/**
 * Current number of connections
 * 
 * @type Number
 */
requestCnt = 0,

/**
 * Queue waiting for connection
 * 
 * @type Array
 */
requestQueue = [],

/**
 * Flag to cancel the `open` command waiting for connection
 * 
 * @type Boolean
 */
requestQueueSkipOpen = false,








/**
 * Protocol version
 *
 * @type String
 **/
this.api = null;

/**
 * elFinder use new api
 *
 * @type Boolean
 **/
this.newAPI = false;

/**
 * elFinder use old api
 *
 * @type Boolean
 **/
this.oldAPI = false;

/**
 * Net drivers names
 *
 * @type Array
 **/
this.netDrivers = [];

/**
 * Base URL of elfFinder library starting from Manager HTML
 * 
 * @type String
 */
this.baseUrl = '';

/**
 * Base URL of i18n js files
 * baseUrl + "js/i18n/" when empty value
 * 
 * @type String
 */
this.i18nBaseUrl = '';

/**
 * Is elFinder CSS loaded
 * 
 * @type Boolean
 */
this.cssloaded = false;

/**
 * Current theme object
 * 
 * @type Object|Null
 */
this.theme = null;

this.mimesCanMakeEmpty = {};

/**
 * Callback function at boot up that option specified at elFinder starting
 * 
 * @type Function
 */
this.bootCallback;

/**
 * Callback function at reload(restart) elFinder 
 * 
 * @type Function
 */
this.reloadCallback;

/**
 * ID. Required to create unique cookie name
 *
 * @type String
 **/
this.id = id;










/**
 * Arrays that has to unbind events
 * 
 * @type Object
 */
this.toUnbindEvents = {};




/**
 * Volume option to set the properties of the root Stat
 * 
 * @type Object
 */
this.optionProperties = {
    icon: void(0),
    csscls: void(0),
    tmbUrl: void(0),
    uiCmdMap: {},
    netkey: void(0),
    disabled: []
};





/**
 * Ajax request type
 *
 * @type String
 * @default "get"
 **/
this.requestType = /^(get|post)$/i.test(this.options.requestType) ? this.options.requestType.toLowerCase() : 'get';

// set `requestMaxConn` by option
requestMaxConn = Math.max(parseInt(this.options.requestMaxConn), 1);

/**
 * Custom data that given as options
 * 
 * @type Object
 * @default {}
 */
this.optsCustomData = $.isPlainObject(this.options.customData) ? this.options.customData : {};

/**
 * Any data to send across every ajax request
 *
 * @type Object
 * @default {}
 **/
this.customData = Object.assign({}, this.optsCustomData);

/**
 * Previous custom data from connector
 * 
 * @type Object|null
 */
this.prevCustomData = null;

/**
 * Any custom headers to send across every ajax request
 *
 * @type Object
 * @default {}
 */
this.customHeaders = $.isPlainObject(this.options.customHeaders) ? this.options.customHeaders : {};

/**
 * Any custom xhrFields to send across every ajax request
 *
 * @type Object
 * @default {}
 */
this.xhrFields = $.isPlainObject(this.options.xhrFields) ? this.options.xhrFields : {};










/**
 * command names for into queue for only cwd requests
 * these commands aborts before `open` request
 *
 * @type Array
 * @default ['tmb', 'parents']
 */
this.abortCmdsOnOpen = this.options.abortCmdsOnOpen || ['tmb', 'parents'];

/**
 * ui.nav id prefix
 * 
 * @type String
 */
this.navPrefix = 'nav' + (elFinder.prototype.uniqueid? elFinder.prototype.uniqueid : '') + '-';

/**
 * ui.cwd id prefix
 * 
 * @type String
 */
this.cwdPrefix = elFinder.prototype.uniqueid? ('cwd' + elFinder.prototype.uniqueid + '-') : '';

// Increment elFinder.prototype.uniqueid
++elFinder.prototype.uniqueid;

/**
 * URL to upload files
 *
 * @type String
 **/
this.uploadURL = opts.urlUpload || opts.url;

/**
 * Events namespace
 *
 * @type String
 **/
this.namespace = namespace;

/**
 * Today timestamp
 *
 * @type Number
 **/
this.today = (new Date(date.getFullYear(), date.getMonth(), date.getDate())).getTime()/1000;

/**
 * Yesterday timestamp
 *
 * @type Number
 **/
this.yesterday = this.today - 86400;






/**
 * elFinder node z-index (auto detect on elFinder load)
 *
 * @type null | Number
 **/
this.zIndex;

/**
 * Current search status
 * 
 * @type Object
 */
this.searchStatus = {
    state  : 0, // 0: search ended, 1: search started, 2: in search result
    query  : '',
    target : '',
    mime   : '',
    mixed  : false, // in multi volumes search: false or Array that target volume ids
    ininc  : false // in incremental search
};

/**
 * Interface language
 *
 * @type String
 * @default "en"
 **/
this.lang = this.storage('lang') || this.options.lang;
if (this.lang === 'jp') {
    this.lang = this.options.lang = 'ja';
}





/**
 * Delay in ms before open notification dialog
 *
 * @type Number
 * @default 500
 **/
this.notifyDelay = this.options.notifyDelay > 0 ? parseInt(this.options.notifyDelay) : 500;

/**
 * Dragging UI Helper object
 *
 * @type jQuery | null
 **/
this.draggingUiHelper = null;

/**
 * Base droppable options
 *
 * @type Object
 **/
this.droppable =







/**
 * Return true if filemanager is active
 *
 * @return Boolean
 **/
this.enabled = function() {
    return enabled && this.visible();
};

/**
 * Return true if filemanager is visible
 *
 * @return Boolean
 **/
this.visible = function() {
    return node[0].elfinder && node.is(':visible');
};

/**
 * Return file is root?
 * 
 * @param  Object  target file object
 * @return Boolean
 */
this.isRoot = function(file) {
    return (file.isroot || ! file.phash)? true : false;
};







/**
 * Registered shortcuts
 *
 * @type Object
 **/
this.shortcuts = function() {
    var ret = [];
    
    $.each(shortcuts, function(i, s) {
	ret.push([s.pattern, self.i18n(s.description)]);
    });
    return ret;
};



/**
 * Supported check hash algorisms
 * 
 * @type Array
 */
self.hashCheckers = [];






/**
 * History object. Store visited folders
 *
 * @type Object
 **/
this.history = new this.history(this);

/**
 * Root hashed
 * 
 * @type Object
 */
this.roots = {};

/**
 * leaf roots
 * 
 * @type Object
 */
this.leafRoots = {};

this.volumeExpires = {};

/**
 * Loaded commands
 *
 * @type Object
 **/
this._commands = {};








/**
 * UI command map of cwd volume ( That volume driver option `uiCmdMap` )
 *
 * @type Object
 **/
this.commandMap = {};

/**
 * cwd options of each volume
 * key: volumeid
 * val: options object
 * 
 * @type Object
 */
this.volOptions = {};

/**
 * Has volOptions data
 * 
 * @type Boolean
 */
this.hasVolOptions = false;

/**
 * Hash of trash holders
 * key: trash folder hash
 * val: source volume hash
 * 
 * @type Object
 */
this.trashes = {};

/**
 * cwd options of each folder/file
 * key: hash
 * val: options object
 *
 * @type Object
 */
this.optionsByHashes = {};

/**
 * UI Auto Hide Functions
 * Each auto hide function mast be call to `fm.trigger('uiautohide')` at end of process
 *
 * @type Array
 **/
this.uiAutoHide = [];


    ============================   Functions   ==============================

/**
 * Store info about files/dirs in "files" object.
 *
 * @param  Array  files
 * @param  String data type
 * @return void
 **/
cache = function(data, type)

/**
 * Delete file object from files caches
 * 
 * @param  Array  removed hashes
 * @return void
 */
remove = function(removed)

/**
 * Update file object in files caches
 * 
 * @param  Array  changed file objects
 * @return void
 */
change = function(changed)

/**
 * Delete cache data of files, ownFiles and self.optionsByHashes
 * 
 * @param  Object  file
 * @param  Boolean update
 * @return void
 */
deleteCache = function(file, update)


/**
 * Exec shortcut
 *
 * @param  jQuery.Event  keydown/keypress event
 * @return void
 */
execShortcut = function(e)

/**
 * Method to store/fetch data
 *
 * @type Function
 **/
this.storage = function()


/**
 * Set pause page unload check function or Get state
 *
 * @param      Boolean   state   To set state
 * @param      Boolean   keep    Keep disabled
 * @return     Boolean|void
 */
this.pauseUnloadCheck = function(state, keep)


/**
 * Configuration options
 *
 * @type Object
 **/
//this.options = $.extend(true, {}, this._options, opts);
this.options = Object.assign({}, this._options);



/**
 * Attach listener to events
 * To bind to multiply events at once, separate events names by space
 * 
 * @param  String  event(s) name(s)
 * @param  Object  event handler or {done: handler}
 * @param  Boolean priority first
 * @return elFinder
 */
this.bind = function(event, callback, priorityFirst)


/**
 * Remove event listener if exists
 * To un-bind to multiply events at once, separate events names by space
 *
 * @param  String    event(s) name(s)
 * @param  Function  callback
 * @return elFinder
 */
this.unbind = function(event, callback)

/**
 * Fire event - send notification to all event listeners
 * In the callback `this` becames an event object
 *
 * @param  String   event type
 * @param  Object   data to send across event
 * @param  Boolean  allow modify data (call by reference of data) default: true
 * @return elFinder
 */
this.trigger = function(evType, data, allowModify)

/**
 * Get event listeners
 *
 * @param  String   event type
 * @return Array    listed event functions
 */
this.getListeners = function(event)


/**
 * Replace XMLHttpRequest.prototype.send to extended function for 3rd party libs XHR request etc.
 * 
 * @type Function
 */
this.replaceXhrSend = function()


/**
 * Restore saved original XMLHttpRequest.prototype.send
 * 
 * @type Function
 */
this.restoreXhrSend = function()


/**
 * Return root dir hash for current working directory
 * 
 * @param  String   target hash
 * @param  Boolean  include fake parent (optional)
 * @return String
 */
this.root = function(hash, fake)


/**
 * Return current working directory info
 * 
 * @return Object
 */
this.cwd = function() {
    return files[cwd] || {};
};


/**
 * Return required cwd option
 * 
 * @param  String  option name
 * @param  String  target hash (optional)
 * @return mixed
 */
this.option = function(name, target)


/**
 * Return disabled commands by each folder
 * 
 * @param  Array  target hashes
 * @return Array
 */
this.getDisabledCmds = function(targets, flip)

/**
 * Return file data from current dir or tree by it's hash
 * 
 * @param  String  file hash
 * @return Object
 */
this.file = function(hash, alsoHidden)


/**
 * Return list of file parents hashes include file hash
 * 
 * @param  String  file hash
 * @return Array
 */
this.parents = function(hash)


/**
 * Return file path or Get path async with jQuery.Deferred
 * 
 * @param  Object  file
 * @param  Boolean i18
 * @param  Object  asyncOpt
 * @return String|jQuery.Deferred
 */
this.path = function(hash, i18, asyncOpt) 


/**
 * Return file url if set
 * 
 * @param  String  file hash
 * @param  Object  Options
 * @return String|Object of jQuery Deferred
 */
this.url = function(hash, o)

/**
 * Return file url for the extarnal service
 *
 * @param      String  hash     The hash
 * @param      Object  options  The options
 * @return     Object  jQuery Deferred
 */
this.forExternalUrl = function(hash, options)

/**
 * Return file url for open in elFinder
 * 
 * @param  String  file hash
 * @param  Boolean for download link
 * @return String
 */
this.openUrl = function(hash, download)


/**
 * Return thumbnail url
 * 
 * @param  Object  file object
 * @return String
 */
this.tmb = function(file)


/**
 * Return selected files hashes
 *
 * @return Array
 **/
this.selected = function() {
    return selected.slice(0);
};

/**
 * Return selected files info
 * 
 * @return Array
 */
this.selectedFiles = function() {
    return $.map(selected, function(hash) { return files[hash] ? Object.assign({}, files[hash]) : null; });
};


/**
 * Return true if file with required name existsin required folder
 * 
 * @param  String  file name
 * @param  String  parent folder hash
 * @return Boolean
 */
this.fileByName = function(name, phash)


/**
 * Valid data for required command based on rules
 * 
 * @param  String  command name
 * @param  Object  cammand's data
 * @return Boolean
 */
this.validResponse = function(cmd, data) {
    return data.error || this.rules[this.rules[cmd] ? cmd : 'defaults'](data);
};


/**
 * Return bytes from ini formated size
 * 
 * @param  String  ini formated size
 * @return Integer
 */
this.returnBytes = function(val)


*****************

/**
 * Process ajax request.
 * Fired events :
 * @todo
 * @example
 * @todo
 * @return $.Deferred
 */
this.request = function(opts)


******************


/**
 * Call cache()
 * Store info about files/dirs in "files" object.
 *
 * @param  Array  files
 * @return void
 */
this.cache = function(dataArray)


/**
 * Update file object caches by respose data object
 * 
 * @param  Object  respose data object
 * @return void
 */
this.updateCache = function(data)


/**
 * Compare current files cache with new files and return diff
 * 
 * @param  Array   new files
 * @param  String  target folder hash
 * @param  Array   exclude properties to compare
 * @return Object
 */
this.diff = function(incoming, onlydir, excludeProps)


/**
 * Sync content
 * 
 * @return jQuery.Deferred
 */
this.sync = function(onlydir, polling)


/**
 * Bind keybord shortcut to keydown event
 *
 * @example
 *    elfinder.shortcut({ 
 *       pattern : 'ctrl+a', 
 *       description : 'Select all files', 
 *       callback : function(e) { ... }, 
 *       keypress : true|false (bind to keypress instead of keydown) 
 *    })
 *
 * @param  Object  shortcut config
 * @return elFinder
 */
this.shortcut = function(s)


/**
 * Get/set clipboard content.
 * Return new clipboard content.
 *
 * @example
 *   this.clipboard([]) - clean clipboard
 *   this.clipboard([{...}, {...}], true) - put 2 files in clipboard and mark it as cutted
 * 
 * @param  Array    new files hashes
 * @param  Boolean  cut files?
 * @return Array
 */
this.clipboard = function(hashes, cut)


/**
 * Return true if command enabled
 * 
 * @param  String       command name
 * @param  String|void  hash for check of own volume's disabled cmds
 * @return Boolean
 */
this.isCommandEnabled = function(name, dstHash)


/**
 * Exec command and return result;
 *
 * @param  String         command name
 * @param  String|Array   usualy files hashes
 * @param  String|Array   command options
 * @param  String|void    hash for enabled check of own volume's disabled cmds
 * @return $.Deferred
 */		
this.exec = function(cmd, files, opts, dstHash)


/**
 * Create and return dialog.
 *
 * @param  String|DOMElement  dialog content
 * @param  Object             dialog options
 * @return jQuery
 */
this.dialog = function(content, options)


/**
 * Create and return toast.
 *
 * @param  Object  toast options - see ui/toast.js
 * @return jQuery
 */
this.toast = function(options)


/**
 * Return UI widget or node
 *
 * @param  String  ui name
 * @return jQuery
 */
this.getUI = function(ui)

/**
 * Return elFinder.command instance or instances array
 *
 * @param  String  command name
 * @return Object | Array
 */
this.getCommand = function(name)


/**
 * Resize elfinder node
 * 
 * @param  String|Number  width
 * @param  String|Number  height
 * @return void
 */
this.resize = function(w, h)


/**
 * Restore elfinder node size
 * 
 * @return elFinder
 */
this.restoreSize = function() {
    this.resize(width, height);
};


/**
 * Lazy execution function
 * 
 * @param  Object  function
 * @param  Number  delay
 * @param  Object  options
 * @return Object  jQuery.Deferred
 */
this.lazy = function(func, delay, opts)


/**
 * Destroy this elFinder instance
 *
 * @return void
 **/
this.destroy = function()


/**
 * Start or stop auto sync
 * 
 * @param  String|Bool  stop
 * @return void
 */
this.autoSync = function(mode)

/**
 * Return bool is inside work zone of specific point
 * 
 * @param  Number event.pageX
 * @param  Number event.pageY
 * @return Bool
 */
this.insideWorkzone = function(x, y, margin)


/**
 * Target ui node move to last of children of elFinder node fot to show front
 * 
 * @param  Object  target    Target jQuery node object
 */
this.toFront = function(target)

/**
 * Remove class 'elfinder-frontmost' and hide() to target ui node
 *
 * @param      Object   target  Target jQuery node object
 * @param      Boolean  nohide  Do not hide
 */
this.toHide =function(target, nohide)


/**
 * Return css object for maximize
 * 
 * @return Object
 */
this.getMaximizeCss = function()


/**
 * Decoding 'raw' string converted to unicode
 * 
 * @param  String str
 * @return String
 */
this.decodeRawString = function(str)


/**
 * Gets target file contents by file.hash
 *
 * @param      String  hash          The hash
 * @param      String  responseType  'blob' or 'arraybuffer' (default)
 * @return     arraybuffer|blob  The contents.
 */
this.getContents = function(hash, responseType)


/**
 * Parse error value to display
 *
 * @param  Mixed  error
 * @return Mixed  parsed error
 */
this.parseError = function(error)


/**
 * Alias for this.trigger('error', {error : 'message'})
 *
 * @param  String  error message
 * @return elFinder
 **/
this.error = function()
