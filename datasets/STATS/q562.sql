select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v,          users as u  where p.Id = pl.PostId 	and p.Id = c.PostId 	and p.Id = ph.PostId 	and p.Id = v.PostId 	and u.Id = p.LastEditorUserId  AND c.CreationDate>='2010-07-28 09:35:29'::timestamp  AND p.Score<=13  AND p.AnswerCount<=8  AND u.Reputation>=1  AND u.Views>=0  AND u.Views<=34  AND u.DownVotes<=11  AND u.UpVotes=0  AND u.CreationDate<='2014-08-14 08:28:47'::timestamp  AND v.CreationDate<='2014-09-09 00:00:00'::timestamp;