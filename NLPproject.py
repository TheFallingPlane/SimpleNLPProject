from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from textblob.classifiers import MaxEntClassifier
import sys



train = [
         ('I love this sandwich.', 'pos'),
         ('this is an amazing place!', 'pos'),
         ('I feel very good about these beers.', 'pos'),
         ('this is my best work.', 'pos'),
         ("what an awesome view", 'pos'),
         ("Black Hawk Down meets Star Wars, that is what it's like watching Rogue One and it truly delivers on the action. This has some of the absolute best action sequences in any Star Wars film and it actually has what many of the films have lacked, stakes. Through good characters and terrific acting, Rogue One manages to succeed what most prequels fail to do, enhance the impact of later films. I see myself watching A New Hope through a new point of view and I have to say, Gareth Edwards made a Star Wars movie that stands on its own. Yes it's not a perfect movie, but it's a really good movie with minor flaws.", 'pos'),
         ("The movie has bit too quick of a start. You are jumping from one planet to another in span of few minutes and can get lost there. However, with second act of the movie this is gone and third one is probably best epic space fight we saw in Star Wars universe ever. The movie does a lot of fanservice, obvious to any SW fan, but which non-fans and first timers would probably didnt even notice (Hammerhead class ship from KoToR), however is funny, quick paced, the plot gets it together in second act and ties up really great at the end, the acting was very well done (especially Donnie Yen). Great movie, go watch, 6.5/10 average by professional critics is absurd.", 'pos'),
         ("Loved it. Was a much better Star Wars movie than the force awakens. This is the Star Wars movie many of us have been waiting for. Darker and more real. Will definitely see it again at the movies.", 'pos'),
         ("One of the best movies of all time. Don't deprive yourself of the pleasure of watching such a stellar movie. Go now and watch this movie if you have never seen it. Perfect casting, amazing story, emotional performances. Just perfect.", 'pos'),
         ("Spectacular!", 'pos'),
         ("I didnt enjoy it as much as I should of, but I know this is a great film.", 'pos'),
         ("This movie is beautiful, genius and brutal. It shows what prison life is like at the time which may not be innovative because obviously it has been done before but it still is a movie that is lightened up by the superb and versatile actings of Tim Robbins (Andy Dufrane) and Morgan Freeman (Redd). The movie is sad and long but shows that if you live in prison you are still just trying to live the life you already had. The movie is a jolly tale of friendship, violence (including torture and isolation) and last but not least genius.", 'pos'),
         ("i cant believe this film is not in the all time top 10 this is a truely great film with a great feel good ending, morgan freeman is immense!", 'pos'),
         ("This movie is possibly the most moving movie I ever watched. I think Morgan Freeman really made this movie what it is, with one of his best performances ever.", 'pos'),
         ("One of the most American movies ever made. A brilliant tale of the American Dream narrated by a simple man with simple pleasures. Hanks is at his stunning best here, and Zemeckis proved that he's more than just a technological wizard when it comes to making movies. A classic that deserves to be adored for decades to come.", 'pos'),
         ("A beautiful movie...by far the best i've seen yet... historically accurate (at least in most places) and has a great ending..though sad Everyone should watch this movie and give credit to its actors...especially Tom Hanks.", 'pos'),
	 ("I loved this movie i thought it was super funny and i liked even though he was slow and ppl made fun of him because of it he still knew how to ignore them and didnt let it bother him he went on with his life and thats how it should be but as a overall rating i liked the movie alot!", 'pos'),
	 ("One of the best movies of all time. A must-see. Tom Hanks is simply fantastic. Everyone must see this film once in their lives, it's a real masterpiece.", 'pos'),
	 ("A great movie! Tom Hanks is an excellent actor and his performance as playing disabled is very good. As we go through Gump`s life, we also go trough modern American history!", 'pos'),
	 ("Good narration of the movie and an in depth feeling brought about in each character. The characters playing around had also been perfectly synchronized with Gump's life like Lt. Dan Taylor, a wonderful acting performance by Tom and Dan.", 'pos'),
	 ("Simply the best acting performance in the history of filmmaking.", 'pos'),
	 ("This captivating look at some of the most profound modern American history and pop-culture demonstrates a true storytelling experience, and expresses the idea that any one man can have a great impact on the world without the knowledge of doing so, and that intelligence is not a lack of wisdom, or of character, and if possible, the child-like innocence captured by Gump's obvious handicaps make him more of a compassionate and loving human being than any lying and big-headed member of society gifted with intelligence.", 'pos'),
	 ("One Of The Greatest Movie and Greatest Book Adaptation ever. Certainly among the Top5 films of the past 2 decades.", 'pos'),
	 ("What a magical movie through the eyes of Forrest Gump. This is a phenomenon of a simple man who changed the world forever.", 'pos'),
	 ("Best movie I have seen. Tom Hanks does excellent job as an actor, but the best part of this movie is the way of telling the story. In addition the whole story tells something about the history of USA. This movie has everything.", 'pos'),
	 ("It's the best movie I have ever seen!!!!", 'pos'),
	 ("This was one of the best movies i have ever seen in my whole life. I laughed, i cried, i was angry i really had real emotions during this movie one of the greatest movies that have ever made.", 'pos'),
	 ("One of the greatest films Tom Hanks has done. I love the humore, the sadnest, the happinest. When it's sad, you can feel the pain that Forrest goes through. Or atleast I can. I always cry whenever someone dies. I can really feel his pain. A really great movie that everyone should see.", 'pos'),
	 ("This movie revolutionized my taste in movies. I was 13 when I first saw this film, and it packed a solid punch that left me in a daze for days on end. Brad Pitt, Edward Norton, and Helena Bonham Carter are all cast to perfection. Pitt in particular is at his personal best. I loved the book by Chuck Pahlaniuk, but I feel that Fincher pumped additional life into the story. It is visionary, innovative, and singular. There have been many attempts to copycat the feeling that this movie created, but it seems like every attempt at touching the greatness of this movie have fallen flat. I don't think that everyone is going to enjoy this movie. I would never watch it with my parents for instance, but it really is great. It was once a cult hit, but it seems to have leaped forward into mainstream pop culture, and I think it deserves the attention.", 'pos'),
	 ("One of the best movies of all times. Of course, the title doesn´t say much, but the story ends up being a wonderful masterpiece. The great antisocial sense only makes it more exciting.", 'pos'),
	 ("A highly underrated film by critics. An excellent film that will keep the viewer grasped in and experience a something completely new. The acting, script, direction, editing and sound are all top notch. There is a lot that can be picked up from this movie, and Fight Club makes you enjoy taking it all in.", 'pos'),
	 ("Fight Club is perhaps one of the most ingenious, pitch perfect films ever made. It hooks you with intense narratives, rocks your world with explosive scenes of extreme intensity, and then shatters your perspective of reality, leaving you at your knees, clapping, and laughing maniacally. It is crafted with the utmost brilliance and is easily one of the greatest films of all time.", 'pos'),
	 ("This movie was amazing, it was hilarious at times, it was gripping, the story was very well done and I was always left anticipating what would happen next, trying to put all the pieces of the non-linear narrative together as a spectator was fun, yes, fun, to do.", 'pos'),
	 ("An excellent adaptation of the book. The small changes are excusable. Characters perform excellently and the story flows and comes together with elegance and class", 'pos'),
	 ("A movie that at first glance looks like some beat-em-up comedy-drama with Brad Pitt, is soooo much more after you watch it. Check it out, although you may have to re-watch it to understand the deeper meaning involved.", 'pos'),
	 ("This Book based movie is a classic. For the person interested in the book, the movie mirrors the book almost identically minus the copy-write changes. This movie does have a man factor to it. However, the metaphor for social injustice is made to a mature aduiance. Its a cult classic!", 'pos'),
	 ("Both dark and hilarious. This film not only delivers great plot of the evolving mind of the narrator (Edward Norton), great performances from Brad Pitt and Edward Norton, it also shows a different view on life and the destruction of the system that we blindly follow.", 'pos'),
	 ("Great movie, some wannabes says its pointless and its but thats just because they want people to think that they know something about movies. This film is genius.", 'pos'),
	 ("it's one of the movies that you get to see once a life-time i loved the story and how anyone can relate to it Edward Norton did an amazing job and it was the first movie that i actually enjoyed brad pitt acting.", 'pos'),
	 ("Slow on the uptake, this film made me a fan of Ed Norton. Obviously the split identity Tyler Durden, played by Pitt, is a perfect outlet for Ed's straight laced character. Call me naive but when I first saw the flick i think I was the last one in the theater to realize the two were the same. Loved the intertwining the making of soap into the storyline as well.", 'pos'),
	 ("One of the best movies i have ever seen. It took me a long time to watch it and when i finally did i can say If you haven't watched it yet, do it as fast as you possibly can.", 'pos'),
	 ("One of the most realistic WW2 movies, Saving Private Ryan has everything; a powerhouse cast, smart action, deep character depth, and a poignant sacrifice.", 'pos'),
	 ("A very high quality, which blends extreme action, with a underlying sadness for the disaster that was world war 2. All put together with superb casting.", 'pos'),
	 ("One of Spielberg's greatest movies! The script is amazing and the the acting is great, also one of Tom Hanks' greatest movies. The violence, action and tragedy in the movie really bring out the realism of the war. I would certainly recommend this movie to anyone that can handle some graphic, painful, bloody action.", 'pos'),
	 ("A true work of art and masterpiece in every sense of the word. Maybe it's just because I'm a fan but I don't think I'm being biased when I say that this is one of the greatest motion pictures in the history of the cinema. The only thing that prevents me from giving it a perfect rating is because I feel the graphic detail of violence, however realistic and brutal reminder of how horrific war can be, leaves no room for imagination. But other than that this is an exceptional film.", 'pos'),
	 ("Another masterpiece from Steven Spielberg! He shows the true face war, humanity and veterans! Amazing acting, superb story and great soundtrack from John Williams makes this movie a MUST WATCH!", 'pos'),
	 ("One of the most powerful war movies out there, Saving will hit you with such an emotion for the characters that it feels like you're in their shoes, in that battle zone, fighting for your life. Tom Hanks is at his best in this movie, and Matt Damon is perfect as Private Ryan. If you don't want to watch all of the movie, at least see the first beach scene. It will change your view of... well, everything. It will make you glad to be alive.", 'pos'),
	 ("Perfect War movie.The first 27 minutes are no less than real. It has everything from emotions to acting.Great direction by critically acclaimed Steven Spielberg and superb acting by Tom Hanks aka Capt. John H. Miller.", 'pos'),
	 ("Greatest war movie I have ever seen. It is powerful, moving and keeps you wanting to watch more. Tom Hanks' and Matt Damon's acting is very good and every one of the characters are deep and you feel like you emphasize with them and understand them. Some scenes bend the laws of physics, but it doesn't make any difference. This movie is not for the faint-hearted as it contains some incredibly gory scenes reminiscent on Saw. Awesome film and should stay on every shelf in every house.", 'pos'),
	 ("Saving Private Ryan has to be an all time favourite for everyone who likes war films. I personally enjoyed the movie because of its gripping action scenes and the very moving storyline which shows the bond between soldiers who have been tasked with a near-suicidal mission. Just simply amazing.", 'pos'),
         ('I do not like this restaurant', 'neg'),
         ('I am tired of this stuff.', 'neg'),
         ("I can't deal with this", 'neg'),
         ('he is my sworn enemy!', 'neg'),
         ('my boss is horrible.', 'neg'),
	 ('Not nearly as funny as it ought to have been, and overly reliant upon stereotypical humor.', 'neg'),
	 ('The ONLY thing good about this movie is the presence of one of America is truly great yet sadly and vastly underrated actors, Ladies and Gentlemen, Mr Ray Liotta. John Travolta was great in Pulp Fiction, but Ray was much, much greater in Goodfellas. Because of his natural intensity, Ray will not get as many parts as some others may, but he still is one of Americas best actors, right up there with Jack, Harvey Keitel, Chris Walken, and Charlie Sheen. (Just kidding about that last one, kids.', 'neg'),
	 ('When I see some of the user reviews here, I can only imagine the studio paid people to post them. This is the most ridiculous movie I have seen in a long time. Oh, comedies should be ridiculous? Not like this. Horrible plot, elements stupid beyond belief (we threatened the gang with a lawsuit, so now I can punch them in the nads?), no funny jokes, poorly executed repetitive slapstick. And one more Hollywood movie thats main selling point is that all men are stupid and secretly half-gay. Any actual biker would find this movie ultimately insulting.', 'neg'),
	 ('Saw this movie on the plane coming back home. Some good laughs (quite few in my opinion) and lots of missed chances. If you are looking for a movie you do not actually have to listen in order to grasp the storyline fairly (I must have fell dosed on it a couple of times and still perfectly understood what was going on), this on should please you, though I know quite a few more I would recommend first.', 'neg'),
	 ('A bunch of washed up middle aged actors trying to make one last decent film. Failure.', 'neg'),
	 ('Mildly funny, wildly homophobic. Wow, the family values are on display here from Disney: Lots of jokes about over-bearing wives controlling their husbands and gays. Homosexuals take a beating here as a daft cop and a prancing thug are the only gay men represented, add in "Deliverence" jokes and this film probably was well-loved by Pat Robertson. I expect Allen, Travolta and Larwrence to do this film, but what is Macy here for? Gosh, he was great in "Fargo."', 'neg'),
	 ('After fifty minutes of suffering through this offensive, trite, unfunny piece of garbage, I walked out. Shame on every last person who had anything to do with releasing this trash. There was nothing even remotely funny, and a waste of fairly good talent. Travolta, Macy, and Allen have each been much better, but with such a poor script, they were totally lost. Pathetic movie-making.', 'neg'),
	 ('This movie is the definition of horrible. Anyone who thinks otherwise is wrong and clearly lacks any abilty to truly gage actual comedy. This is not funny. I would not see it again if I was paid $50. I would pay $250 to have never seen it. Probably the worst mistake I will make this year. Do not see this movie!!!!!', 'neg'),
	 ('I really do not see how anybody can enjoy movies like this. Movies that were put together in a few minutes, with no thought put into it. Not funny at all, and a completely unconvincing story. If you know what makes a good movie, you agree with me. If the stupidest things take your fancy, then you disagree with me, and like this movie. You wanna see a funny movie? Watch Pineapple Express, or Pulp Fiction. Both those movies over shadow this movie in every single facet.', 'neg'),
	 ('I gave in to my girlfriends wishes to see this movie. I would seen the trailers, which were awful, and watching the movie in its entirety was just brutal. Lame, astoundingly unfunny, and boring. Was this script written in five minutes?! Two hours of the Weather Channel would offer more laughs.', 'neg'),
	 ('The begining was a little funny but other then that its all downhill from there, Its boring and just another generic road trip comedy that does nothing but give a few pathetic laughs.', 'neg'),
	 ('Painfully stupid. The movie is a clear result of what happens when you try to bringing actors together for the sake of bringing actors together instead of actually formulating a worthwhile script.', 'neg'),
	 ('This movie is stupid to a level I can barely comprehend. The funny moments are few to none, and the amount of pure idiocy shown in a brigade of adult men in a travesty.', 'neg'),
	 ('Wild Hogs thinks that it is so wild and it thinks they have hogs as well to call themselves the Wild Hogs. Now let me give you two words to explain this aroused hog found in the ground. Not Fun.', 'neg'),
	 ('I am sorry I wasted my time with this film. I was excited for the potential in this type of fantasy/modern mashup but the writing throughout was flat out awful. It seemed like it was written in one go, and you can literally see the characters barf nonsense exposition at each moment where the writers needed to figure out what to do next. Like watching two six year olds play fantasy cops in the backyard. And I dunno, maybe that s what they were going for, but I was expecting something with a little thought put into it, instead this is an Ed Wood or Troma level B movie. With higher paid actors? Whatever, forget it.', 'neg'),
	 ('Great premise bad execution. Will Smith is amazing as usual but that is not enough to save this movie. Lets not give bad movies a pass just because of their good premise.', 'neg'),
	 ('As always ... Racism, racism, racism. Only feminists and transgender people are missing! How long will this continue?', 'neg'),
	 ('Same basic concept as alien nation just with a predictable and cliche story that was poorly executed.', 'neg'),
	 ('What an offensive, ridiculous mess. Between the nonsensical racial allegory that trades in stereotypes like they are Pokemon cards and the dark, unwatchable images on screen, it is easy to see why critics panned this appalling monstrosity. What is unclear is why Netflix ordered a sequel.', 'neg'),
	 ('This movie is bad. It’s slow, unoriginal, and ruins a great premise. So if you have Netflix, skip it.', 'neg'),
	 ('Every mistake you can make in producing a movie, this movie made. There is WAAAAY too much expository dialogue, all of it written in SO abruptly and unnatural. This world does not feel unnatural because there are fantasy creatures in it--it feels unnatural because ALL of the dialogue does not flow or feel like someone would ever actually say any of the things characters are saying. Unwatchable.', 'neg'),
	 ('When a movie has you begging for it to end not even half way through it is pure crap. We haveve all seen this movie and this characters millions of times, nothing new in it. Do not waste your time.', 'neg'),
	 ('Maybe I am not english so I missed the jokes - but it was not romantic or comedic. An hour in I was bored to shreds .... Kiera Knightly was the only interesting thing for me in this movie.... but worth sitting 2hrs for without being delved into the story.', 'neg'),
	 ('Watched it again, and this time I realized just how glitzy and poor this film is...Really appealed to my baser instincts when I first watched it....Everything warm and fuzzy at Christmas with a couple of "real" love stories...But beneath the coating of Hollywood style slickness, there lurks mediocrity, a lust substitute for love...Firth likes his maids butt, Rickman loves to ogle his come on sexy secretary, and those were mild examples...A film of mostly male fantasies done in a "cute" simplistic way.....What a waste of talented actors....Liam Neeson was decent....And Linneys story is about love...Beneath the coating of sparkles and sleaze, lurks a mediocre film....Merry Christmas!', 'neg'),
	 ('Such a mediocre movie! I can not believe how people say that over and over again over Christmas. So many stereotypes -- gay British rock stars, exotic innocent foreign beauty, messed-up 10 Downing St., foolish American girls completely infatuated by a random British guy. By the way how can the only black in this movie cheated on and the only mentally ill guy seem to be a mere burden on family?', 'neg'),
	 ('I hope the next movie will be Santa Clause- The Death Clause. Totally awful movie. The acting, the story, and the directing were sub par. I dont recommend you waste your time or money on this movie.', 'neg'),
	 ('Total CRAP!! After seeing this movie, I have lost all respect for Tim Allen and Martin Short. The only way you would possibly like this movie is if you were 3 years old. Terrible acting! I can not believe I wasted 6 bucks on it! CRAP!', 'neg'),
	 ('This movie was bad from all angles. 1. the story really never started until the last 30 minutes. 2. the musical score was not even christmas style music and they only used one christmas song that I can remember in the soundtrack. 3. There was no Scott Calvin sarcasm. This whole movie was slapstick. Not what we fell in love with in the first movie. 4. There was no reason for Mrs. Clause to be pregnant. There was no reason for Scott Calvin to bring his exwife and her husband to the north pole. they were useless characters and not apart of the story. And worst of all the #1 elf was so annoying with his lisp that I wanted to punch him. The should have paid the money it cost to bring Bernard back for the role. Do not waste your money. Not even at matinee prices. The story was written so poorly that I was angry when I left and not in the christmas spirit. Not one thing made me laugh. I laughed so hard in the first one.', 'neg'),
	 ('Lame, boring and Martin Short deserves a Razzie for worst irritating actor.', 'neg'),
	 ('It is Christmas season so obviously, Tim Allen has to milk the holiday for all that it is worth by putting out another lame, derivative, pointless movie.', 'neg'),
	 ('Thank god for Martin Short, as he is the only saving grace in this sinking film franchise. But even he can not save the ship, because it has too many holes that can never be filled up, even if the movie tries to pile on more and more gags and slapstick humor.', 'neg'),
	 ('This movie is so boring and dumb that to me, I would not even recommend to people who celebrate Christmas. I have not seen the other films, but I probably wont because there probably just as bad as this film that does not feel like a film.', 'neg'),
	 ('Totally awful movie. The acting, the story, and the directing were sub par. I do not recommend you waste your time or money on this movie. So you had better be off seeing the first two movies instead.', 'neg'),
	 ('I remember getting free tickets for this film and I came out so disappointed. Its seems so full of itself and its done in a bad carry on Troying way. The death scenes seem to be hammed up and the eptiomy of how cheesy this film is when Brad Pitt says "Take it, its yours". I like epic war films, histroical war films. This is bloody awful, and I think it only made money because of the stature of certain actors. Gladiator and Braveheart are brilliant, this one is the complete opposite. The only reason it gets a 3 is because of some of the designs..and thats about it.', 'neg'),
	 ('This movie is bad. I read the book and the real story is way better. Brad pitt is the worst actor.', 'neg'),
	 ('Extremely long, very boring and tried way too hard. The battle scenes were not that great. Terrible acting to go around and a script that seem to be made by morons. Just like the Trojans were. Do not be fooled by the spectacle.', 'neg'),
	 ('Stanley Kubrick, a famous director, turns into a dirty old man before he dies. This movie is probably about his own fantasies. It is just sickening. What is the point of this movie ? It is not even interesting. Every man has fantasies about sexual experiences with other women, and on a daily basis. What else is new ?', 'neg'),
	 ('This 13-year gap means that we were expecting such a film to be great, especially whenever it comes to Thorntons character stealing the show back in 2003. Only this time around, he manages to have a bit of changes because not only Bad Santa 2 manages to fill the lump of coal this early, it also manages to be naughty terrible for a combined reasons why.', 'neg'),
	 ('Not much to laugh about here it is just a overly crude overly dumb mess. The first film while crude had a sharp scathing sense of humor. This film just throws a sex joke then a racist joke then a handicap joke at the wall and very little sticks. By the way Thurman is one of the most annoying characters of the year.', 'neg'),
	 ('The very few bits that I found funny through out the run time do not justify seeing it at all. It is a nostalgia trip that does nothing to even come close to the epic original, Which is a Christmas classic imo. It is a needless sequel that should have happened much sooner than it did and by the time it came out, No one cared.', 'neg'),
	 ('Slow and boring. First half of the movie is spoken all in Indian. Not my cup of tea. I think the lead actor is a great actor. But i just could not sit threw this boring film.', 'neg'),
	 ('The only tears were tears of excruciating boredom. Sure, it is a nice true human story that shows cultures many may be unfamiliar with. Yes it is shot well in exotic and far off locations (from a domestic standpoint) featuring some decent acting. Who cares when it is this low key and this familiar a story? Not this viewer for one.', 'neg'),
	 ('Failed movie and still a Fail.', 'neg'),
	 ('No. I cannot even express the failure this movie had. This is a waste of money, and a disgrace for a movie. Extreme fail. I feel bad for whoever created this movie.', 'neg'),
	 ('Another movie with an extremely rare 1 Metascore. What do I have to say about this movie? There is nothing interesting about this movie. What is the point of watching it? Go play a board game, read a book, do homework, heck... even watch a whole load of movies. If you watch this, you will be bored until you sleep.', 'neg'),
	 ('I hated this movie.', 'neg')
	 
        ]

continueOption = 'y'
movieNames = ['2001: A Space Odyssey' , 'Pirates of the Caribbean: Dead Man Tell No Tales' , 'The Shawshank Redemption' , 'Captain America: Civil War' , 'Avengers: Infinity War']
expectedPosResult = [ 50 , 20 , 70 , 60 , 40]
expectedNegResult = [ 50 , 80 , 30 , 40 , 60]

classifierNaiveBayes = NaiveBayesClassifier(train)
classifierDecisionTree = DecisionTreeClassifier(train)





while (continueOption == 'y'):

    posCount = 0
    negCount = 0
    maxPosPercent = 0
    maxNegPercent = 0
    
    posCountNaiveBayes = 0
    posCountDecisionTree = 0
    posCountMaxEnt = 0

    negCountNaiveBayes = 0
    negCountDecisionTree = 0
    negCountMaxEnt = 0

    neutCountNaiveBayes = 0
    neutCountDecisionTree = 0
    neutCountMaxEnt = 0
    
    totalCount = 30

    posPercentNaiveBayes = 0
    posPercentDecisionTree = 0
    posPercentMaxEnt = 0

    neutPercentNaiveBayes = 0
    neutPercentDecisionTree = 0
    neutPercentMaxEnt = 0
    
    negPercentNaiveBayes = 0
    negPercentDecisionTree = 0
    negPercentMaxEnt = 0
    

    maxPosPercentNaiveBayes = 0
    maxPosPercentDecisionTree = 0
    maxPosPercentMaxEnt = 0
    
    maxNegPercentNaiveBayes = 0
    maxNegPercentDecisionTree = 0
    maxNegPercentMaxEnt = 0

    maxPosIndexNaiveBayes = 0
    maxPosIndexDecisionTree = 0
    maxPosIndexMaxEnt = 0
    
    maxNegIndexNaiveBayes = 0
    maxNegIndexDecisionTree = 0
    maxNegIndexMaxEnt = 0

    option = input('Please choose a movie. \n 1-2001: A Space Odyssey \n 2-Pirates of the Carribean: Dead Man Tell No Tales \n 3-The Shawshank Redemption \n 4-Captain America: Civil War \n 5-Avengers: Infinity War \n 6-Movie with the most positive comments \n 7-Movie with the most negative comments \n Your input here =  ')
    movieNo = int(option)

    if(option == '6'):
        
        for i in range(1,6):
            
            for j in range(1,31):
                
                fileName = ('Movies\M%d\%d.txt' % (i,j))
                file = open(fileName, 'r')
                contents = file.read()

                

                if(classifierNaiveBayes.classify(contents) == 'pos'):
                    posCount +=1
                    
                elif(classifierNaiveBayes.classify(contents) == 'neg'):
                    negCount +=1
                    

            posPercent = (posCount / totalCount) * 100

            if(maxPosPercent < posPercent):
                maxPosPercent = posPercent
                maxPosIndex = i
                
            posCount = 0
            
            
        print('%s has the most positive comments with %d percent.' % (movieNames[maxPosIndex-1],maxPosPercent))

        continueOption = input('Do you want to choose another film? (y/n) = ')

        
    elif(option == '7'):
        
        for i in range(1,6):
            
            for j in range(1,31):
                
                fileName = ('Movies\M%d\%d.txt' % (i,j))
                file = open(fileName, 'r')
                contents = file.read()


                if(classifierNaiveBayes.classify(contents) == 'pos'):
                    posCount +=1
                    
                elif(classifierNaiveBayes.classify(contents) == 'neg'):
                    negCount +=1
                    

            negPercent = (negCount / totalCount) * 100

            if(maxNegPercent < negPercent):
                maxNegPercent = negPercent
                maxNegIndex = i
                
            negCount = 0
            
            
        print('%s has the most negative comments with %d percent.' % (movieNames[maxNegIndex-1],maxNegPercent))

        continueOption = input('Do you want to choose another film? (y/n) = ')

        
    else:
        
        for i in range(1,31):
            
            fileName = ('Movies\M%d\%d.txt' % (movieNo,i))
            file = open(fileName, 'r')
            contents = file.read()


            if (classifierNaiveBayes.classify(contents) == 'neg'):
                negCountNaiveBayes +=1

            elif (classifierNaiveBayes.classify(contents) == 'pos'):
                posCountNaiveBayes +=1
            else:
                neutCountNaiveBayes+=1


            if (classifierDecisionTree.classify(contents) == 'neg'):
                negCountDecisionTree +=1

            elif (classifierDecisionTree.classify(contents) == 'pos'):
                posCountDecisionTree +=1
            else:
                neutCountDecisionTree+=1






        negPercentNaiveBayes = (negCountNaiveBayes / totalCount) * 100
        posPercentNaiveBayes = (posCountNaiveBayes / totalCount) * 100

        negPercentDecisionTree = (negCountDecisionTree / totalCount) * 100
        posPercentDecisionTree = (posCountDecisionTree / totalCount) * 100

        negPercentMaxEnt = (negCountMaxEnt / totalCount) * 100


        neutPercentNaiveBayes = (neutCountNaiveBayes / totalCount) * 100
        neutPercentDecisionTree = (neutCountDecisionTree / totalCount) * 100


    
        print('Comment rates for %s are below.\n\n' % (movieNames[movieNo-1]))

        print('Expected rates for %s are PosRate = %d NegRate = %d\n\n' % (movieNames[movieNo-1] , expectedPosResult[movieNo-1] , expectedNegResult[movieNo-1]))
        
        print ('Positive Comment Rate With Naive Bayes Classifier = %d' % (posPercentNaiveBayes))
        print ('Negative Comment Rate With Naive Bayes Classifier = %d' % (negPercentNaiveBayes))
        print ('Neutral Comment Rate With Naive Bayes Classifier = %d\n\n' % (neutPercentNaiveBayes))

        print('Positive Comment Rate With Decision Tree Classifier = %d' % (posPercentDecisionTree))
        print('Negative Comment Rate With Decision Tree Classifier = %d' % (negPercentDecisionTree))
        print ('Neutral Comment Rate With Decision Tree Classifier = %d\n\n' % (neutPercentDecisionTree))



        

        continueOption = input('Do you want to choose another film? (y/n) = ')





        
        
        
        







        

        
    
