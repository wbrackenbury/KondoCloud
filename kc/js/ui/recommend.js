/**
 * @class elfinderrecommend - elFinder container for FM recommendations
 * @author Will Brackenbury
 **/

var notLocked = true;

// $.fn.animateHighlight = function (highlightColor, duration) {
//   var highlightBg = highlightColor || "#FFFF9C";
//   var animateMs = duration || 1500;
//   var originalBg = this.css("backgroundColor");
//   if (notLocked) {
//     notLocked = false;
//     this.stop().css("background-color", highlightBg).animate({ backgroundColor: originalBg }, animateMs);
//     setTimeout(function () {
//       notLocked = true;
//     }, animateMs);
//   }
// };

$.fn.elfinderrecommend = function (fm) {
  "use strict";
  var cl = "elfinder-recommend";

  this.not("." + cl).each(function () {
    var wz = $(this).addClass(cl),
      prevH = Math.round(wz.height()),
      parent = wz.parent(),
      setDelta = function () {
        wdelta = wz.outerHeight(true) - wz.height();
      },
      fitsize = function (e) {
        wz.height($(".elfinder-workzone").height());
      },
      cssloaded = function () {
        wdelta = wz.outerHeight(true) - wz.height();
        fitsize();
      },
      // getGradientColor = function (start_color, end_color, percent) {
      //   // strip the leading # if it's there
      //   start_color = start_color.replace(/^\s*#|\s*$/g, "");
      //   end_color = end_color.replace(/^\s*#|\s*$/g, "");

      //   // convert 3 char codes --> 6, e.g. `E0F` --> `EE00FF`
      //   if (start_color.length == 3) {
      //     start_color = start_color.replace(/(.)/g, "$1$1");
      //   }

      //   if (end_color.length == 3) {
      //     end_color = end_color.replace(/(.)/g, "$1$1");
      //   }

      //   // get colors
      //   var start_red = parseInt(start_color.substr(0, 2), 16),
      //     start_green = parseInt(start_color.substr(2, 2), 16),
      //     start_blue = parseInt(start_color.substr(4, 2), 16);

      //   var end_red = parseInt(end_color.substr(0, 2), 16),
      //     end_green = parseInt(end_color.substr(2, 2), 16),
      //     end_blue = parseInt(end_color.substr(4, 2), 16);

      //   // calculate new color
      //   var diff_red = end_red - start_red;
      //   var diff_green = end_green - start_green;
      //   var diff_blue = end_blue - start_blue;

      //   diff_red = (diff_red * percent + start_red).toString(16).split(".")[0];
      //   diff_green = (diff_green * percent + start_green).toString(16).split(".")[0];
      //   diff_blue = (diff_blue * percent + start_blue).toString(16).split(".")[0];

      //   // ensure 2 digits by color
      //   if (diff_red.length == 1) {
      //     diff_red = "0" + diff_red;
      //   }
      //   if (diff_green.length == 1) {
      //     diff_green = "0" + diff_green;
      //   }
      //   if (diff_blue.length == 1) {
      //     diff_blue = "0" + diff_blue;
      //   }

      //   return "#" + diff_red + diff_green + diff_blue;
      // },
      // getcolor = function (colornum) {
      //   var colortotal = 10,
      //     start = "#000000",
      //     end = "#7d7d7d";
      //   return getGradientColor(start, end, 1 - colornum / colortotal);
      // },
      formatrec = function (r) {
        var modr = Object.assign({}, r);

        if (r.ppath === "") {
          modr.modpath = "/";
        } else {
          modr.modpath = r.ppath;
        }

        if (r.name.length > 30) {
          modr.modname = r.name.substring(0, 30) + "...";
        } else {
          modr.modname = r.name;
        }

        if (r.explain_name.length > 30) {
          modr.modexplain = r.explain_name.substring(0, 30) + "...";
        } else {
          modr.modexplain = r.explain_name;
        }

        if (r.dstname && r.dstname.length > 30) {
          modr.moddst = r.dstname.substring(0, 30) + "...";
        } else if (r.dstname === "") {
          modr.moddst = "/";
        } else {
          modr.moddst = r.dstname;
        }

        return modr;
      },
      osopenparent = function (e, node, org, alsoclick, isdst) {
        var rt = $(node.parentNode.parentNode.parentNode).data();
        if (["folderorgclicks", "regclicks"].includes(org)) {
          if (alsoclick) {
            openparentplus(e, rt, isdst);
          } else {
            openparent(e, rt, isdst);
          }
        }
      },
      openparent = function (e, rt, isdst) {
        e.stopPropagation();
        var tosend = rt.parhash;
        if (isdst) {
          tosend = rt.dst;
        }
        fm.request({
          data: { cmd: "open", reload: 0, target: tosend, tree: 0 },
        });
      },
      openparentplus = function (e, rt, isdst) {
        openparent(e, rt, isdst);
        setTimeout(function () {
          $("#" + rt.phash).trigger("click");
        }, 300);
      },
      delparent = function (recbox, delfunc) {
        return function (e) {
          var rt = $(this.parentNode.parentNode.parentNode).data();
          e.stopPropagation();
          //recbox.removeChild(this.parentNode.parentNode);
          fm.request({
            data: {
              cmd: "update_recs",
              rec_ids: [rt.rec_id],
              status: "deleted",
            },
          });
          $(this.parentNode.parentNode.parentNode).fadeOut(200);
          $(this.parentNode.parentNode.parentNode).remove();
          //delfunc(rt.id);
        };
      },
      accmoveparent = function (recbox, delfunc) {
        return function (e) {
          var rt = $(this.parentNode.parentNode.parentNode).data();
          e.stopPropagation();
          $(this.parentNode.parentNode.parentNode).fadeOut(200);
          $(this.parentNode.parentNode.parentNode).remove();
          var somedfrd = fm.request({
            data: { cmd: "getfile", target: rt.phash },
          });
          somedfrd.done(function (data) {
            fm.clipboard([]);
            fm.clipboard([rt.phash], true);
            fm.exec("paste", [rt.dst, rt.phash], {
              _userAction: true,
              _cmd: "move",
              cut: 1,
              rec_id: rt.rec_id,
            });
            //fm.request({data: {cmd: 'paste', targets: rt.phash, dst: rt.dst, cut: 1, rec_id: rt.rec_id}});
          });
          //delfunc(rt.id);
        };
      },
      accdelparent = function (recbox, delfunc) {
        return function (e) {
          var rt = $(this.parentNode.parentNode.parentNode).data();
          e.stopPropagation();
          $(this.parentNode.parentNode.parentNode).fadeOut(200);
          $(this.parentNode.parentNode.parentNode).remove();
          fm.request({
            data: { cmd: "rm", targets: [rt.phash], rec_id: rt.rec_id },
          });
          //delfunc(rt.id);
        };
      },
      accinvrec = function (recbox, delfunc) {
        return function (e) {
          var rt = $(this.parentNode.parentNode.parentNode).data();
          e.stopPropagation();
          $(this.parentNode.parentNode.parentNode).fadeOut(200);
          $(this.parentNode.parentNode.parentNode).remove();

          fm.getCommand("undo").rec_exec(rt.rec_id);

          //fm.request({data: {cmd: 'paste', rec_id: rt.rec_id, targets: rt.phash, dst: rt.parhash, cut: 1, inv: 1}});
          //delfunc(rt.id);
        };
      },
      explain = function (e, node, explain_type) {
        var rt = $(node.parentNode).data();
        var expl = $("<div id='rec-hover-tool' class='elfinder-rec-explain'></div>");
        var topoffset = 15 + (explain_type === "move" ? 40 : 20);
        var leftoffset = 0;
        var preamble = "You are receiving this recommendation because you";
        if (explain_type === "move") {
          expl.html(`${preamble} moved <b>${rt.explain_name}</b> to this location`);
        } else if (explain_type === "find") {
          expl.html(`${preamble} previewed <b>${rt.explain_name}</b>`);
        } else {
          expl.html(`${preamble} deleted <b>${rt.explain_name}</b>`);
        }
        $("body").append(expl);

        expl.css({
          top: $(node).offset().top + topoffset,
          left: $(node).offset().left + leftoffset,
        });
      },
      delexplain = function (e, node) {
        $("#rec-hover-tool").remove();
      },
      /**

        Creates a single row in a recommendation

        Arguments:

          fronttext (str): Text to appear at beginning of row
          modr (Object): modified recommendation object
          orgstyle (str): Organization style of recommendations
          clickstyle (bool): Whether we should highlight a file when
                            clicking a hover
          isdst (bool): Send to destination or to parent

      */
      recrow = function (fronttext, backtext, type_row, orgstyle, clickstyle, isdst, fulltext) {
        var inline = $("<div class='elfinder-new-rec-row'></div>");
        var inspan = $(`<span class="elfinder-new-rec-row-left-label">${fronttext}</span>`);
        var inpart = $(`<span class='elfinder-name-hover'>${backtext}</span>`);
        inpart = inpart.on("click", function (e) {
          osopenparent(e, this, orgstyle, clickstyle, isdst);
        });

        if (fulltext) {
          inpart.attr("title", fulltext);
        }

        inline.append(inspan);
        inline.append(inpart);

        return inline;
      },
      addTooltip = function (target, type_row) {
        target
          .on("mouseover", function (e) {
            e.stopPropagation();
            explain(e, this, type_row);
          })
          .on("mouseout", function (e) {
            e.stopPropagation();
            delexplain(e, this);
          });
      },
      buttons = function (recbox, all, invert, typebutt) {
        var buttonclasses = ["elfinder-rec-butts", "ui-state-default", "elfinder-button"];
        var accbuttclass = Object.assign([], buttonclasses);
        var rejbuttclass = Object.assign([], buttonclasses);
        var buttcontainer = $("<div class='elfinder-new-butt-container'></div>");
        var accrecicon = $("<span class='elfinder-button-icon-accept'></span>");
        var refrecicon = $("<span class='elfinder-button-icon-clear'></span>");
        var invrecicon = $("<span class='elfinder-button-icon-invert'></span>");
        var accrec;
        var refrec;
        var invrec;
        var toptext_one;
        var toptext_two;

        if (all) {
          accrec = $(`<div tooltip='Accept recommendation' class="${accbuttclass.join(" ")}"></div>`).on(
            "click",
            allacc(recbox)
          );
          refrec = $(`<div tooltip='Clear recommendation' class="${rejbuttclass.join(" ")}"></div>`).on(
            "click",
            alldelclose(recbox)
          );
          if (typebutt === "find") {
            toptext_one = $("<span class='elfinder-rec-all-button-text'>Clear all</span>");
          } else {
            toptext_one = $("<span class='elfinder-rec-all-button-text'>Accept all</span>");
            toptext_two = $("<span class='elfinder-rec-all-button-text'>Clear all</span>");
          }
          buttcontainer.append(toptext_one);
        } else {
          accbuttclass.push("elfinder-acc-butt");
          rejbuttclass.push("elfinder-rej-butt");

          if (typebutt === "find") {
            accrec = $(""); // No accept button for find recommendations
            refrec = $(
              "<div tooltip='Clear recommendation' class='" + rejbuttclass.join(" ") + "'></div>"
            ).on("click", delparent(recbox, fm.removeFindRec));
          } else {
            if (invert) {
              invrec = $(
                "<div tooltip='Undo recommendation' class='" + buttonclasses.join(" ") + "'></div>"
              ).on("click", accinvrec(recbox, fm.invRec));
              accrec = $("");
              refrec = $("");
              invrec.append(invrecicon);
            } else {
              if (typebutt === "move") {
                accrec = $(
                  `<div
                      tooltip='Accept recommendation'
                      class="${accbuttclass.join(" ")}"></div>`
                ).on("click", accmoveparent(recbox, fm.removeAndInvRec));
              } else if (typebutt === "del") {
                accrec = $(
                  `<div
                      tooltip='Accept recommendation'
                      class="${accbuttclass.join(" ")}"></div>`
                ).on("click", accdelparent(recbox, fm.removeAndInvRec));
              }
              refrec = $(
                `<div tooltip='Clear recommendation'
                  class="${rejbuttclass.join(" ")}"></div>`
              ).on("click", delparent(recbox, fm.removeRec));
            }
          }
        }

        accrec.append(accrecicon);

        if (typebutt !== "find") {
          if (invert) {
            buttcontainer.append(invrec);
          } else {
            buttcontainer.append(accrec);
          }
        }
        if (all) {
          if (typebutt !== "find") {
            buttcontainer.append(toptext_two);
          }
          refrec.append(refrecicon);
          buttcontainer.append(refrec);
        }

        return buttcontainer;
      },
      delsec = function (recbox) {
        var delsc = $("<div class='elfinder-rec-del-sec'></div>"),
          rejbuttclass = [
            "elfinder-rec-butts",
            "ui-state-default",
            "elfinder-rec-small-button",
            "elfinder-rej-butt",
          ],
          refrecicon = $("<div class='elfinder-button-icon-exit'></div>").on("mouseover", function (e) {
            $(this).addClass("elfinder-button-icon-exit-hover");
          }),
          refrec;

        refrec = $(`<div class="${rejbuttclass.join(" ")}"></div>`).on(
          "click",
          delparent(recbox, fm.removeFindRec)
        );
        refrec.append(refrecicon);
        delsc.append(refrec);
        return delsc;
      },
      singlefindrec = function (r, orgstyle) {
        var recbox = $(".elfinder-recommend-find")[0];
        var modr = formatrec(r);
        // var color = getcolor(r.num);
        var fullrec = $(`<div id="${r.rec_id}" class='elfinder-new-rec'></div>`);
        var windowdel = delsec(recbox);
        var findline = recrow("Find:", modr.modname, "find", orgstyle, true, false, modr.name);
        var inline = recrow("In:", modr.modpath, "find", orgstyle, false, false, modr.ppath);
        var buttcontainer = buttons(recbox, false, false, "find");

        var buttonContainer = $("<div class='elfinder-rec-button-container'></div>");
        buttonContainer.append(windowdel);
        buttonContainer.append(buttcontainer);

        var rectext = $("<div class='elfinder-new-rec-content'></div>");
        rectext.append(findline);
        rectext.append(inline);
        addTooltip(rectext, "find");

        fullrec.append(buttonContainer);
        fullrec.append(rectext);
        fullrec.data(modr);

        $(".elfinder-recommend-find .empty-recs-msg");

        return fullrec;
      },
      allacc = function (recbox) {
        return function (e) {
          e.stopPropagation();
          $([recbox])
            .find(".elfinder-acc-butt")
            .each(function () {
              $(this).trigger("click");
            });
        };
      },
      alldelclose = function (recbox) {
        return function (e) {
          e.stopPropagation();
          $([recbox])
            .find(".elfinder-rej-butt")
            .each(function () {
              $(this).trigger("click");
            });
        };
      },
      singlemoverec = function (r, i, orgstyle) {
        var focus_file_isdst = !!r.inv;
        var modr = formatrec(r);
        var recbox = $(".elfinder-recommend-move")[0];
        // var color = getcolor(r.num);
        var fullrec = $(`<div id="${r.rec_id}" class='elfinder-new-rec'></div>`);
        var windowdel = delsec(recbox);

        var moveline = recrow("Move:", modr.modname, "move", orgstyle, true, focus_file_isdst, modr.name);
        var fromline = recrow("From:", modr.modpath, "move", orgstyle, false, false, modr.ppath);
        var toline = recrow("To:", modr.moddst, "move", orgstyle, false, true, modr.dstname);
        var buttcontainer = buttons(recbox, false, modr.inv, "move");

        // if (!r.inv) {
        //   fullrec.css("color", color);
        // }

        if (i == 0 && ["folderorg", "folderorgclicks"].includes(orgstyle)) {
          $(recbox).append(`<div class='elfinder-rec-header'>Located in <i>${modr.modpath}</i></div>`);
        }

        var rectext = $("<div class='elfinder-new-rec-content'></div>");
        rectext.append(moveline);
        rectext.append(fromline);
        rectext.append(toline);
        addTooltip(rectext, "move");

        var buttonContainer = $("<div class='elfinder-rec-button-container'></div>");
        buttonContainer.append(buttcontainer);
        buttonContainer.append(windowdel);

        fullrec.append(buttonContainer);
        fullrec.append(rectext);
        fullrec.data(modr);

        $(".elfinder-recommend-move .empty-recs-msg").remove();

        return fullrec;
      },
      singledelrec = function (r, i, orgstyle) {
        var recbox = $("#removeheader")[0];
        var modr = formatrec(r);
        // var color = getcolor(r.num);
        var fullrec = $(`<div id="${r.rec_id}" class='elfinder-new-rec'></div>`);
        var windowdel = delsec(recbox);
        var moveline = recrow("Delete:", modr.modname, "del", orgstyle, true, false, modr.name);
        var fromline = recrow("From:", modr.modpath, "del", orgstyle, false, false, modr.ppath);
        var buttcontainer = buttons(recbox, false, modr.inv, "del");

        // if (!r.inv) {
        //   fullrec.css("color", color);
        // }

        if (i == 0 && ["folderorg", "folderorgclicks"].includes(orgstyle)) {
          $(recbox).append(`<div class='elfinder-rec-header'>Located in <b>${modr.modpath}</b></div>`);
        }

        var buttonContainer = $("<div class='elfinder-rec-button-container'></div>");
        buttonContainer.append(buttcontainer);
        buttonContainer.append(windowdel);

        var rectext = $("<div class='elfinder-new-rec-content'></div>");
        rectext.append(moveline);
        rectext.append(fromline);
        addTooltip(rectext, "remove");

        fullrec.append(buttonContainer);
        fullrec.append(rectext);
        fullrec.data(modr);

        $(".elfinder-recommend-move .empty-recs-msg").remove();

        return fullrec;
      },
      recfind = function (e) {
        var recs = fm.getFindRecs();
        var found = e.data.found;
        var recbox = $(".elfinder-recommend-find")[0];
        var orgstyle = fm.getRecStyle();
        var child = recbox.lastElementChild;
        while (child) {
          recbox.removeChild(child);
          child = recbox.lastElementChild;
        }

        var i, r, modpath, fulltop, topspan, topbutts, isdone;

        findheaderprep();

        var recs_arr = recs,
          to_del = [];

        for (i = 0; i < recs_arr.length; i++) {
          r = recs_arr[i];
          isdone = r.phash === found;
          if (isdone) {
            r.status = "done_no_acc";
            to_del.push(r);
            continue;
          }

          var fullrec = singlefindrec(r, orgstyle);
          $(recbox).append(fullrec);
        }

        // for (var j = 0; j < to_del.length; j++) {
        // fm.request({data: {cmd: 'update_recs',
        //                    rec_ids: [to_del[j].rec_id],
        //                    status: to_del[j].status}});
        // }
      },
      findheaderprep = function () {
        headerprep(".elfinder-recommend-find", "Potentially Relevant Files", "find");
      },
      moveheaderprep = function () {
        headerprep(".elfinder-recommend-move", "Suggested File Moves", "move");
      },
      removeheaderprep = function () {
        headerprep(".elfinder-recommend-remove", "Suggested File Deletes", "move");
      },
      headerprep = function (recboxname, headertext, type) {
        var recbox = $(recboxname)[0];
        var fulltop = $("<div class='elfinder-rec-label'></div>");
        fulltop.append($(`<span class='elfinder-rec-topline-span'>${headertext}</span>`));
        fulltop.append(buttons(recbox, true, false, type));
        $(recbox).append(fulltop);
      },
      recomendTopHeader = function () {
        $($("#recheader")[0]).append($("<div class='rec-header'>Recommendations</div>"));
      },
      logbutton = function (wz) {
        var butt = $(`<div class='elfinder-recommend-accordion rec-section-head'>
          <div class="rec-section-head"> Accepted Recommendations Log</div>
          <div id='reclog' class='elfinder-recommend-log rec-section-head'></div>
        </div>`);
        // var log = $("<div id='reclog' class='elfinder-recommend-log rec-section-head'></div>");

        butt.on("click", function (e) {
          $(".elfinder-recommend")[0].classList.toggle("accordion-active");
        });
        wz.append(butt);
        // wz.append(log);
        return butt;
      },
      renderinvrecs = function (e) {
        var recs = fm.getRecs();
        var removed = e.data.removed;
        var recbox = $("#reclog")[0];
        var orgstyle = fm.getRecStyle();
        var alreadydonefunc = function (rt) {
          return function (a) {
            return rt.phash === a;
          };
        };
        var child = recbox.lastElementChild;
        var to_del = [];
        var isdone;
        while (child) {
          recbox.removeChild(child);
          child = recbox.lastElementChild;
        }

        var i, r, fullrec;

        for (i = 0; i < recs.length; i++) {
          r = recs[i];

          if (!r.inv) {
            continue;
          }

          if (r.action === "move") {
            fullrec = singlemoverec(r, i, orgstyle);
          } else if (r.action === "del") {
            fullrec = singledelrec(r, i, orgstyle);
          } else {
            fullrec = singlefindrec(r, orgstyle);
          }
          $(recbox).append(fullrec);
        }
      },
      recmove = function (e) {
        var recs = fm.getRecs();
        var removed = e.data.removed;
        var recbox = $("#moveheader")[0];
        var orgstyle = fm.getRecStyle();
        var child = recbox.lastElementChild;
        while (child) {
          recbox.removeChild(child);
          child = recbox.lastElementChild;
        }

        moveheaderprep();

        var recs_arr = recs,
          l = recs_arr.length,
          to_del = [],
          alreadydonefunc = function (rt) {
            return function (a) {
              return rt.phash === a;
            };
          },
          i,
          r,
          modname,
          modpath,
          moddst,
          color,
          isdone;

        /**
                       If one of the items that just got moved is already
                       something that we're recommending they do,
                       we should delete the recommendation since it's
                       now completed. cf. alreadydonefunc, isdone
                    */

        var dfd = $.Deferred();

        for (i = 0; i < l; i++) {
          r = recs_arr[i];

          if (r.inv || r.action !== "move") {
            continue;
          }

          // isdone = (removed.findIndex(alreadydonefunc(r)) != -1);

          // if (isdone) {
          //     r.status = "done_no_acc";
          //     to_del.push(r);
          //     continue;
          // }

          var fullrec;

          fullrec = singlemoverec(r, i, orgstyle);

          $(recbox).append(fullrec);
        }

        renderinvrecs(e);
      },
      recdel = function (e) {
        var recs = fm.getRecs();
        var removed = e.data.removed;
        var recbox = $("#removeheader")[0];
        var orgstyle = fm.getRecStyle();
        var child = recbox.lastElementChild;
        while (child) {
          recbox.removeChild(child);
          child = recbox.lastElementChild;
        }

        removeheaderprep();

        var recs_arr = recs,
          l = recs_arr.length,
          to_del = [],
          alreadydonefunc = function (rt) {
            return function (a) {
              return rt.phash === a;
            };
          },
          i,
          r,
          modname,
          modpath,
          moddst,
          color,
          isdone;

        /**
                       If one of the items that just got moved is already
                       something that we're recommending they do,
                       we should delete the recommendation since it's
                       now completed. cf. alreadydonefunc, isdone
                    */

        var dfd = $.Deferred();

        for (i = 0; i < l; i++) {
          r = recs_arr[i];

          if (r.inv || r.action !== "del") {
            continue;
          }

          // isdone = (removed.findIndex(alreadydonefunc(r)) != -1);

          // if (isdone) {
          //     r.status = "done_no_acc";
          //     to_del.push(r);
          //     continue;
          // }

          var fullrec;

          fullrec = singledelrec(r, i, orgstyle);

          $(recbox).append(fullrec);
        }

        renderinvrecs(e);
      },
      wdelta;

    setDelta();
    parent.on("resize." + fm.namespace, fitsize);
    wz.append("<div id='recheader' class=''></div>");
    wz.append(`<div id='moveheader' class='elfinder-recommend-move rec-section-head'></div>`);
    wz.append(`<div id='removeheader' class='elfinder-recommend-remove rec-section-head'></div>`);
    wz.append(`<div id='findheader' class='elfinder-recommend-find rec-section-head'></div>`);
    logbutton(wz);
    recomendTopHeader();
    moveheaderprep();
    removeheaderprep();
    findheaderprep();
    const emptyMsg = `<div class='empty-recs-msg'>
                No recomendations of this type yet! Keep reorganizing to get automated suggestions.
              </div>`;
    // fulltop.append(
    ["#moveheader", "#removeheader", "#findheader"].forEach((id) => {
      $(id).append($(emptyMsg));
    });
    // );
    fm.one("cssloaded", cssloaded)
      .bind("uiresize", fitsize)
      .bind("themechange", setDelta)
      .bind("paste", recmove)
      .bind("find", recfind)
      .bind("undo", recmove)
      .bind("undo", recdel)
      .bind("redo", recmove)
      .bind("redo", recdel)
      .bind("rm", recdel);
  });
  return this;
};
