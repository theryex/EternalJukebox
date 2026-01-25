
function createJRemixer(context) {
    var remixer = {

        remixTrack: function (track, canonizerData, callback) {

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
                            track.status = 'ok'
                            callback(1, track, 100);
                        },
                        function (e) { // error function
                            track.status = 'error: loading audio'
                            callback(-1, track, 0);
                            console.log('audio error', e);
                            error("trouble loading audio");
                        }
                    );
                };

                request.onerror = function (e) {
                    trace('error loading loaded');
                    track.status = 'error: loading audio'
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
                        var qlist = track.analysis[type]

                        j = parseInt(j)

                        var q = qlist[j]
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
                    var qparent = qparents[i]
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
                    var q = quanta[i]

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
            fetchAudio(canonizerData.audioURL === null ? "api/audio/jukebox/" + track.info.id : ("api/audio/external?fallbackID=" + track.info.id + "&url=" + encodeURIComponent(canonizerData.audioURL)));
        },

        getPlayer: function () {
            var curQ = null;
            var curAudioSource = null;
            var masterGain = .53;
            var deltaTime = 0;
            var mainGain = context.createGain();
            var otherGain = context.createGain();
            var skewDelta = 0;
            var maxSkewDelta = .05;
            var ocurAudioSource = null;

            mainGain.connect(context.destination);
            otherGain.connect(context.destination);

            mainGain.gain.value = masterGain;
            otherGain.gain.value = 1 - masterGain;

            function playQuantumWithDurationSimple(when, q, dur, gain) {
                var start = 0;

                if (dur == undefined) {
                    dur = q.duration;
                }

                if (gain == undefined) {
                    gain = 1;
                }

                var duration = dur;

                var audioSource = context.createBufferSource();
                var audioGain = ('createGain' in context) ? context.createGain() : context.createGainNode();
                audioGain.gain.value = gain;
                audioSource.buffer = q.track.buffer;
                audioSource.connect(audioGain);
                audioSource.start(start, q.start, duration);
                audioGain.connect(context.destination);
                return duration + when;
            }

            function llPlay(buffer, start, duration, gain) {
                var audioSource = context.createBufferSource();
                audioSource.buffer = buffer;
                audioSource.connect(gain);
                audioSource.start(0, start, duration);
                return audioSource;
            }


            function playQ(q) {
                // all this complexity is about click reduction.
                // We want to continuously play as much as we can
                // without getting out of sync
                if (curQ == null || curQ.next != q) {
                    if (curAudioSource) {
                        curAudioSource.stop();
                    }
                    var tduration = q.track.audio_summary.duration - q.start;
                    curAudioSource = llPlay(q.track.buffer, q.start, tduration, mainGain);
                    deltaTime = context.currentTime - q.start;
                }

                var now = context.currentTime - deltaTime;
                var delta = now - q.start;

                otherGain.gain.value = (1 - masterGain) * q.otherGain;
                if (curQ == null || curQ.other.next != q.other || Math.abs(skewDelta) > maxSkewDelta) {
                    skewDelta = 0;
                    if (ocurAudioSource) {
                        ocurAudioSource.stop();
                    }
                    var oduration = q.other.track.audio_summary.duration - q.other.start;
                    ocurAudioSource = llPlay(q.other.track.buffer, q.other.start, oduration, otherGain);
                }
                skewDelta += q.duration - q.other.duration;
                curQ = q;
                return q.duration - delta;
            }

            var player = {
                play: function (when, q, duration, gain) {
                    return playQuantumWithDurationSimple(when, q, duration, gain);
                },

                playQ: function (q) {
                    return playQ(q);
                },

                stop: function () {
                    if (curAudioSource) {
                        curAudioSource.stop(0);
                        curAudioSource = null;
                    }
                    if (ocurAudioSource) {
                        ocurAudioSource.stop(0);
                        ocurAudioSource = null;
                    }
                },

                curTime: function () {
                    return context.currentTime;
                }
            }
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
