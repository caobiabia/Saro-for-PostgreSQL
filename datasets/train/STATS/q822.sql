select  count(*) from comments as c,  		posts as p,  		votes as v,         users as u  where p.Id = c.PostId 	and p.Id = v.PostId 	and u.id = p.LastEditorUserId  AND p.CommentCount>=0  AND u.Views>=0  AND u.Views<=24  AND u.DownVotes>=0  AND u.CreationDate>='2010-09-15 09:37:05'::timestamp  AND u.CreationDate<='2014-09-12 09:36:40'::timestamp  AND v.CreationDate>='2010-07-22 00:00:00'::timestamp  AND v.CreationDate<='2014-09-13 00:00:00'::timestamp;