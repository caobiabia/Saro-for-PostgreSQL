select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v,          users as u  where p.Id = pl.PostId 	and p.Id = c.PostId 	and p.Id = ph.PostId 	and p.Id = v.PostId 	and u.Id = p.LastEditorUserId  AND pl.CreationDate>='2011-11-15 22:50:47'::timestamp  AND p.PostTypeId=2  AND p.ViewCount>=0  AND p.ViewCount<=11691  AND p.CreationDate>='2010-07-19 21:23:20'::timestamp  AND u.Reputation<=126  AND u.Views>=0  AND u.Views<=1305  AND u.CreationDate<='2014-09-10 21:50:38'::timestamp  AND v.CreationDate>='2010-08-06 00:00:00'::timestamp  AND v.CreationDate<='2014-09-09 00:00:00'::timestamp;