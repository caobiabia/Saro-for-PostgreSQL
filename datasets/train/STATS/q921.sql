select  count(*) from comments as c,  		postLinks as pl,  		postHistory as ph,          votes as v where pl.PostId = c.PostId 	and c.PostId = ph.PostId 	and ph.PostId = v.PostId  AND c.CreationDate>='2010-07-29 21:53:37'::timestamp  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-21 00:00:00'::timestamp  AND v.CreationDate<='2014-09-13 00:00:00'::timestamp;