select  count(*) from postLinks as pl,          posts as p,  		users as u,  		badges as b  where p.Id = pl.RelatedPostId  	and u.Id = p.OwnerUserId 	and u.Id = b.UserId  AND pl.CreationDate>='2010-10-01 23:13:10'::timestamp  AND pl.CreationDate<='2014-08-22 22:02:12'::timestamp  AND p.Score<=39  AND p.CommentCount<=8  AND p.CreationDate>='2010-07-20 14:38:07'::timestamp  AND u.Reputation<=164  AND u.CreationDate<='2014-08-29 13:17:19'::timestamp;
