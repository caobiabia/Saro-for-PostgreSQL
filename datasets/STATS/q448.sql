select  count(*) from comments as c,  		posts as p,          postLinks as pl,          votes as v,          badges as b,          users as u  where p.Id = c.PostId 	and p.Id = pl.RelatedPostId     and p.Id = v.PostId  	and u.Id = p.LastEditorUserId     and u.Id = b.UserId  AND c.CreationDate>='2010-08-12 20:42:55'::timestamp  AND p.Score>=0  AND p.ViewCount>=0  AND p.ViewCount<=29151  AND p.CommentCount<=11  AND u.Reputation>=1  AND u.CreationDate>='2010-07-27 12:10:34'::timestamp  AND u.CreationDate<='2014-08-28 21:46:20'::timestamp  AND v.BountyAmount>=0  AND v.BountyAmount<=50  AND v.CreationDate<='2014-09-13 00:00:00'::timestamp;