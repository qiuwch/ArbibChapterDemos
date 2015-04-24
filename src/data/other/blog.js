MathJax.Hub.Config({
	tex2jax: {
		inlineMath: [['$','$'], ['\\(','\\)']],
		processEscapes: true
	}
});

$(document).ready(function() {
	hljs.initHighlightingOnLoad(); // start highlight plugin
	
	// Dynamically generate the table of content of document
	// Make this flow and follow the scroll bar
	var nav = $('<nav></nav>');
	var h1 = $('<b>Table of contents</b>');
	var ul = $('<ul></ul>');

	$('#TOC').append(nav);
	nav.append(h1);
	nav.append(ul);

	$("article h2").each(function() {
		// This code is inspired by https://css-tricks.com/automatic-table-of-contents/
		var element = $(this);
		var title = element.text();
		var id = element.attr("id");

		if (id === undefined) {
			id = title.toLowerCase().split(' ').join('-');
			element.attr("id", id);
		}
		var link = "#" + id;

		var li = $('<li></li>');
		var a = $('<a>').text(title).attr('href', link);

		li.append(a);
		ul.append(li);
	});
});