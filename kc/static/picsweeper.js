$(function () {
  
  window.current_offset = 0;
  window.batchsize = 20;
  
  var getPics = function () {
    var params = {start: window.current_offset, count: batchsize};
    $.getJSON("/pictures/page",params,function(body) {
      for (offset in body) {
        $("tbody").append("<tr id='" + body[offset].path_lower + "'><td><input type='checkbox' name='selected_picture' value='" +
          body[offset].path_lower + "'></td><td><img src='" + body[offset].url +
          "'></td><td>" + body[offset].tags + "</td></tr>");
      }
      window.current_offset += body.length;
    });
  };

  $("#picture_selection").submit(function(){
    var selected_rows = $(this).find(":checked").closest("tr");
    $.post($(this).attr('action'),$(this).serialize(),function(response){
      $('#confirm-submit').modal('hide');
      selected_rows.addClass("danger");
      selected_rows.find(":checked").attr("disabled", true);

      console.log(selected_rows);
    });
    return false;
  });

  $("#showmore").click( getPics);
  var stream_url = $("meta[name=stream_url]").attr("content");
  var csrf_token = $("meta[name=csrf_token]").attr("content");
  $.ajaxSetup({
    beforeSend: function (xhr, settings) {
      if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
        xhr.setRequestHeader("X-CSRFToken", csrf_token);
      }
    }
  });
  
  // this script is deferred - stream_url had better be set!)
	var source = new EventSource(stream_url);
	source.addEventListener('status', function(event) {
			var data = JSON.parse(event.data);
      $('#drano-progress-description').text(data.message);
	}, false);
  
  source.addEventListener("batchready", function(event) {
    // this event gets sent when new pics have been fully analyzed by the back end.
    // for now, all it is used for is to pull the first batch of pics dynamically.
    var data = JSON.parse(event.data);
    if (window.current_offset == 0)
      getPics();
  },false);

  source.addEventListener("batchdeleted", function(event) {
    var data = JSON.parse(event.data);
    for (offset in data){
      $("tr#" + CSS.escape(data[offset])).remove();
    }
  }, false);

	
  source.addEventListener('error', function(event) {
    console.log("Failed to connect to event stream. Is Redis running?");
	}, false);

  // if this runs and returns zero, the 'first' listener should still work.
  getPics();
});
