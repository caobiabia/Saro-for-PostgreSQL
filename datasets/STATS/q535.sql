select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v,          users as u  where p.Id = pl.PostId 	and p.Id = c.PostId 	and p.Id = ph.PostId 	and p.Id = v.PostId 	and u.Id = p.LastEditorUserId  AND c.Score=0  AND c.CreationDate>='2010-07-20 05:29:24'::timestamp  AND p.ViewCount>=0  AND u.Reputation>=1  AND u.DownVotes>=0  AND u.UpVotes>=0  AND u.CreationDate<='2014-09-03 21:18:45'::timestamp;