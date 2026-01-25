
function createJRemixer(context) {
    var remixer = {

        remixTrack: function (track, jukeboxData, callback) {

            function fetchAudio(url) {
                var request = new XMLHttpRequest();
                trace("fetchAudio " + url);
                track.buffer = null;
                request.open("GET", url, true);
                request.responseType = "arraybuffer";
                this.request = request;

                request.onload = function () {
                    trace('audio loaded');
                    context.decodeAudioData(request.response,
                        function (buffer) {      // completed function
                            track.buffer = buffer;
                            track.status = 'ok';
                            callback(1, track, 100);
                        },
                        function (e) { // error function
                            track.status = 'error: loading audio';
                            callback(-1, track, 0);
                            document.querySelector("span#info").textContent = "Audio could not be retrieved, provide a youtube link or file on the tune settings"
                            console.log('audio error', e);
                        }
                    );
                };

                request.onerror = function (e) {
                    trace('error loading loaded');
                    track.status = 'error: loading audio';
                    callback(-1, track, 0);
                };

                request.onprogress = function (e) {
                    var percent = Math.round(e.loaded * 100 / e.total);
                    callback(0, track, percent);
                };
                request.send();
            }

            function preprocessTrack(track) {
                trace('preprocessTrack');
                var types = ['sections', 'bars', 'beats', 'tatums', 'segments'];

                for (var i in types) {
                    var type = types[i];
                    trace('preprocessTrack ' + type);
                    for (var j in track.analysis[type]) {
                        var qlist = track.analysis[type];

                        j = parseInt(j);

                        var q = qlist[j];
                        q.track = track;
                        q.which = j;
                        if (j > 0) {
                            q.prev = qlist[j - 1];
                        } else {
                            q.prev = null
                        }

                        if (j < qlist.length - 1) {
                            q.next = qlist[j + 1];
                        } else {
                            q.next = null
                        }
                    }
                }

                connectQuanta(track, 'sections', 'bars');
                connectQuanta(track, 'bars', 'beats');
                connectQuanta(track, 'beats', 'tatums');
                connectQuanta(track, 'tatums', 'segments');

                connectFirstOverlappingSegment(track, 'bars');
                connectFirstOverlappingSegment(track, 'beats');
                connectFirstOverlappingSegment(track, 'tatums');

                connectAllOverlappingSegments(track, 'bars');
                connectAllOverlappingSegments(track, 'beats');
                connectAllOverlappingSegments(track, 'tatums');


                filterSegments(track);
            }

            function filterSegments(track) {
                var threshold = .3;
                var fsegs = [];
                fsegs.push(track.analysis.segments[0]);
                for (var i = 1; i < track.analysis.segments.length; i++) {
                    var seg = track.analysis.segments[i];
                    var last = fsegs[fsegs.length - 1];
                    if (isSimilar(seg, last) && seg.confidence < threshold) {
                        fsegs[fsegs.length - 1].duration += seg.duration;
                    } else {
                        fsegs.push(seg);
                    }
                }
                track.analysis.fsegments = fsegs;
            }

            function isSimilar(seg1, seg2) {
                var threshold = 1;
                var distance = timbral_distance(seg1, seg2);
                return (distance < threshold);
            }

            function connectQuanta(track, parent, child) {
                var last = 0;
                var qparents = track.analysis[parent];
                var qchildren = track.analysis[child];

                for (var i in qparents) {
                    var qparent = qparents[i];
                    qparent.children = [];

                    for (var j = last; j < qchildren.length; j++) {
                        var qchild = qchildren[j];
                        if (qchild.start >= qparent.start
                            && qchild.start < qparent.start + qparent.duration) {
                            qchild.parent = qparent;
                            qchild.indexInParent = qparent.children.length;
                            qparent.children.push(qchild);
                            last = j;
                        } else if (qchild.start > qparent.start) {
                            break;
                        }
                    }
                }
            }

            // connects a quanta with the first overlapping segment
            function connectFirstOverlappingSegment(track, quanta_name) {
                var last = 0;
                var quanta = track.analysis[quanta_name];
                var segs = track.analysis.segments;

                for (var i = 0; i < quanta.length; i++) {
                    var q = quanta[i];

                    for (var j = last; j < segs.length; j++) {
                        var qseg = segs[j];
                        if (qseg.start >= q.start) {
                            q.oseg = qseg;
                            last = j;
                            break
                        }
                    }
                }
            }

            function connectAllOverlappingSegments(track, quanta_name) {
                var last = 0;
                var quanta = track.analysis[quanta_name];
                var segs = track.analysis.segments;

                for (var i = 0; i < quanta.length; i++) {
                    var q = quanta[i];
                    q.overlappingSegments = [];

                    for (var j = last; j < segs.length; j++) {
                        var qseg = segs[j];
                        // seg starts before quantum so no
                        if ((qseg.start + qseg.duration) < q.start) {
                            continue;
                        }
                        // seg starts after quantum so no
                        if (qseg.start > (q.start + q.duration)) {
                            break;
                        }
                        last = j;
                        q.overlappingSegments.push(qseg);
                    }
                }
            }

            preprocessTrack(track);
            fetchAudio(jukeboxData.audioURL === null ? "api/audio/jukebox/" + track.info.id : ("api/audio/external?fallbackID=" + track.info.id + "&url=" + encodeURIComponent(jukeboxData.audioURL)));
        },


        getPlayer: function () {
            var audioGain = context.createGain();
            var curAudioSource = null;
            var curQ = null;
            var curOffset = 0;
            audioGain.gain.value = 0.5;
            audioGain.connect(context.destination);

            function playQuantum(when, q) {
                var now = context.currentTime;
                var start = when === 0 ? now : when;
                var next = start + q.duration;
                var qOffset = q.track.offset || 0;

                if (curAudioSource && curQ && curQ.track === q.track && curQ.which + 1 === q.which && curOffset === qOffset) {
                    // let it ride
                } else {
                    var audioSource = null;
                    var offset = q.start + qOffset;
                    var duration = track.audio_summary.duration - q.start;
                    if (offset + duration > 0) {
                        if (offset < 0) {
                            start += -offset;
                            duration -= -offset;
                            offset = 0;
                        }
                        audioSource = context.createBufferSource();
                        //audioGain.gain.value = 1;
                        audioSource.buffer = q.track.buffer;
                        audioSource.connect(audioGain);
                        audioSource.start(start, offset, duration);
                    }
                    if (curAudioSource) {
                        curAudioSource.stop(start);
                    }
                    curAudioSource = audioSource;
                }
                q.audioSource = curAudioSource;
                curQ = q;
                curOffset = qOffset;
                return next;
            }

            var player = {
                audioGain: audioGain,

                play: function (when, q) {
                    return playQuantum(when, q);
                },

                stop: function (q) {
                    if (q === undefined) {
                        if (curAudioSource) {
                            curAudioSource.stop(0);
                            curAudioSource = null;
                        }
                    } else {
                        if ('audioSource' in q) {
                            if (q.audioSource !== null) {
                                q.audioSource.stop(0);
                            }
                        }
                    }
                    curQ = null;
                },

                curTime: function () {
                    return context.currentTime;
                }
            };
            return player;
        }
    };

    function trace(text) {
        if (false) {
            console.log(text);
        }
    }

    return remixer;
}


function euclidean_distance(v1, v2) {
    var sum = 0;
    for (var i = 0; i < 3; i++) {
        var delta = v2[i] - v1[i];
        sum += delta * delta;
    }
    return Math.sqrt(sum);
}

function timbral_distance(s1, s2) {
    return euclidean_distance(s1.timbre, s2.timbre);
}
