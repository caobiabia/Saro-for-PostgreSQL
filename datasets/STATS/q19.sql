select  count(*) from comments as c,  		posts as p,           postHistory as ph,  		badges as b,          users as u  where u.Id = c.UserId 	and u.Id = p.LastEditorUserId 	and u.Id = ph.UserId 	and u.Id = b.UserId  AND b.Date>='2010-07-20 05:29:09'::timestamp  AND b.Date<='2014-08-07 01:16:09'::timestamp  AND c.Score=0  AND c.CreationDate>='2010-07-19 19:56:21'::timestamp  AND ph.CreationDate<='2014-08-28 06:31:03'::timestamp  AND p.AnswerCount=2  AND p.FavoriteCount<=21  AND u.Views>=0  AND u.Views<=649  AND u.DownVotes>=0  AND u.DownVotes<=1  AND u.UpVotes<=57  AND u.CreationDate>='2010-09-30 17:47:26'::timestamp;