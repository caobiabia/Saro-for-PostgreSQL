select  count(*) from comments as c,          postLinks as pl,          posts as p,  		users as u,  		badges as b   where p.Id = pl.RelatedPostId 	and p.Id = c.PostId 	and u.Id = b.UserId 	and u.Id = p.OwnerUserId  AND b.Date>='2010-07-19 20:14:08'::timestamp  AND c.Score=0  AND c.CreationDate>='2010-07-27 17:10:56'::timestamp  AND c.CreationDate<='2014-09-13 10:58:30'::timestamp  AND pl.CreationDate<='2014-09-08 12:56:29'::timestamp  AND p.PostTypeId=1  AND p.Score>=-4  AND p.Score<=35  AND p.CommentCount<=10  AND u.DownVotes>=0  AND u.DownVotes<=6;